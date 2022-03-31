#include "ParticleSystem.hpp"

#include "MPI/CommMPI.hpp"
#include "Util/EquatnHelper.hpp"
#include "Util/GeoUtil.hpp"
#include "Util/IOHelper.hpp"
#include "Util/Logger.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include <vtkCellData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkTypeInt32Array.h>
#include <vtkTypeUInt8Array.h>
#include <vtkXMLPPolyDataReader.h>
#include <vtkXMLPolyDataReader.h>

#include <mpi.h>
#include <omp.h>

template <int spectralDegree>
ParticleSystem<spectralDegree>::ParticleSystem(const std::string &configFile, const std::string &posFile, int argc, char **argv) {
    initialize(ParticleConfig(configFile), posFile, argc, argv);
}

template <int spectralDegree>
ParticleSystem<spectralDegree>::ParticleSystem(const ParticleConfig &runConfig_, const std::string &posFile, int argc, char **argv) {
    initialize(runConfig_, posFile, argc, argv);
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::initialize(const ParticleConfig &runConfig_, const std::string &posFile, int argc, char **argv) {
    runConfig = runConfig_;
    stepCount = 0;
    snapID = 0; // the first snapshot starts from 0 in writeResult

    // store the random seed
    restartRngSeed = runConfig.rngSeed;

    // set MPI
    int mpiflag;
    MPI_Initialized(&mpiflag);
    TEUCHOS_ASSERT(mpiflag);

    Logger::set_level(runConfig.logLevel);
    commRcp = getMPIWORLDTCOMM();

    showOnScreenRank0();

    // TRNG pool must be initialized after mpi is initialized
    rngPoolPtr = std::make_shared<TRngPool>(runConfig.rngSeed);
    conSolverPtr = std::make_shared<ConstraintSolver>();
    conCollectorPtr = std::make_shared<ConstraintCollector>();

    dinfo.initialize(); // init DomainInfo
    setDomainInfo();

    particleContainer.initialize();
    particleContainer.setAverageTargetNumberOfSampleParticlePerProcess(200); // more sample for better balance

    if (IOHelper::fileExist(posFile)) {
        setInitialFromFile(posFile);
    } else {
        setInitialFromConfig();
    }
    setLinkMapFromFile(posFile);

    // at this point all particles located on rank 0
    commRcp->barrier();
    decomposeDomain();
    exchangeParticle(); // distribute to ranks, initial domain decomposition

    particleNearDataDirectoryPtr = std::make_shared<ZDD<ParticleNearEP>>(particleContainer.getNumberOfParticleLocal());

    treeParticleNumber = 0;
    setTreeParticle();

    calcVolFrac();

    if (commRcp->getRank() == 0) {
        IOHelper::makeSubFolder("./result"); // prepare the output directory
        writeBox();
    }

    if (!runConfig.particleFixed) {
        // 100 NON-B steps to resolve initial configuration collisions
        // no output
        spdlog::warn("Initial Collision Resolution Begin");
        for (int i = 0; i < runConfig.initPreSteps; i++) {
            prepareStep();
            calcVelocityNonCon();
            resolveConstraints();
            saveForceVelocityConstraints();
            sumForceVelocity();
            stepEuler();
        }
        spdlog::warn("Initial Collision Resolution End");
    }

    spdlog::warn("ParticleSystem Initialized. {} local particles", particleContainer.getNumberOfParticleLocal());
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::reinitialize(const ParticleConfig &runConfig_, const std::string &restartFile, int argc,
                                  char **argv, bool eulerStep) {
    runConfig = runConfig_;

    // Read the timestep information and pvtp filenames from restartFile
    std::string pvtpFileName;
    std::ifstream myfile(restartFile);

    myfile >> restartRngSeed;
    myfile >> stepCount;
    myfile >> snapID;
    myfile >> pvtpFileName;

    // increment the rngSeed forward by one to ensure randomness compared to previous run
    restartRngSeed++;

    // set MPI
    int mpiflag;
    MPI_Initialized(&mpiflag);
    TEUCHOS_ASSERT(mpiflag);

    Logger::set_level(runConfig.logLevel);
    commRcp = getMPIWORLDTCOMM();

    showOnScreenRank0();

    // TRNG pool must be initialized after mpi is initialized
    rngPoolPtr = std::make_shared<TRngPool>(restartRngSeed);
    conSolverPtr = std::make_shared<ConstraintSolver>();
    conCollectorPtr = std::make_shared<ConstraintCollector>();

    dinfo.initialize(); // init DomainInfo
    setDomainInfo();

    particleContainer.initialize();
    particleContainer.setAverageTargetNumberOfSampleParticlePerProcess(200); // more samples for better balance

    std::string asciiFileName = pvtpFileName;
    auto pos = asciiFileName.find_last_of('.');
    asciiFileName.replace(pos, 5, std::string(".dat")); // replace '.pvtp' with '.dat'
    pos = asciiFileName.find_last_of('_');
    asciiFileName.replace(pos, 1, std::string("Ascii_")); // replace '_' with 'Ascii_'

    std::string baseFolder = getCurrentResultFolder();
    setInitialFromVTKFile(baseFolder + pvtpFileName);

    setLinkMapFromFile(baseFolder + asciiFileName);

    // VTK data is wrote before the Euler step, thus we need to run one Euler step below
    if (eulerStep)
        stepEuler();

    stepCount++;
    snapID++;

    // at this point all particles located on rank 0
    commRcp->barrier();
    applyBoxBC();
    decomposeDomain();
    exchangeParticle(); // distribute to ranks, initial domain decomposition
    updateParticleMap();

    particleNearDataDirectoryPtr = std::make_shared<ZDD<ParticleNearEP>>(particleContainer.getNumberOfParticleLocal());

    treeParticleNumber = 0;
    setTreeParticle();
    calcVolFrac();

    spdlog::warn("ParticleSystem Initialized. {} local particles", particleContainer.getNumberOfParticleLocal());
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::setTreeParticle() {
    // initialize tree
    // always keep tree max_glb_num_ptcl to be twice the global actual particle number.
    const int nGlobal = particleContainer.getNumberOfParticleGlobal();
    if (nGlobal > 1.5 * treeParticleNumber || !treeParticleNearPtr) {
        // a new larger tree
        treeParticleNearPtr.reset();
        treeParticleNearPtr = std::make_unique<TreeParticleNear>();
        treeParticleNearPtr->initialize(2 * nGlobal);
        treeParticleNumber = nGlobal;
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::getOrient(Equatn &orient, const double px, const double py, const double pz, const int threadId) {
    Evec3 pvec;
    if (px < -1 || px > 1) {
        pvec[0] = 2 * rngPoolPtr->getU01(threadId) - 1;
    } else {
        pvec[0] = px;
    }
    if (py < -1 || py > 1) {
        pvec[1] = 2 * rngPoolPtr->getU01(threadId) - 1;
    } else {
        pvec[1] = py;
    }
    if (pz < -1 || pz > 1) {
        pvec[2] = 2 * rngPoolPtr->getU01(threadId) - 1;
    } else {
        pvec[2] = pz;
    }

    // px,py,pz all random, pick uniformly in orientation space
    if (px != pvec[0] && py != pvec[1] && pz != pvec[2]) {
        EquatnHelper::setUnitRandomEquatn(orient, rngPoolPtr->getU01(threadId), rngPoolPtr->getU01(threadId),
                                          rngPoolPtr->getU01(threadId));
        return;
    } else {
        orient = Equatn::FromTwoVectors(Evec3(0, 0, 1), pvec);
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::setInitialFromConfig() {
    // this function init all particles on rank 0
    if (runConfig.particleLengthSigma > 0) {
        rngPoolPtr->setLogNormalParameters(runConfig.particleLength, runConfig.particleLengthSigma);
    }

    if (commRcp->getRank() != 0) {
        particleContainer.setNumberOfParticleLocal(0);
    } else {
        const double boxEdge[3] = {runConfig.initBoxHigh[0] - runConfig.initBoxLow[0],
                                   runConfig.initBoxHigh[1] - runConfig.initBoxLow[1],
                                   runConfig.initBoxHigh[2] - runConfig.initBoxLow[2]};
        const double minBoxEdge = std::min(std::min(boxEdge[0], boxEdge[1]), boxEdge[2]);
        const double maxLength = minBoxEdge * 0.5;
        const double radius = runConfig.particleDiameter / 2;
        const int nParticleLocal = runConfig.particleNumber;
        particleContainer.setNumberOfParticleLocal(nParticleLocal);

#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < nParticleLocal; i++) {
                // initailize the particle
                double length;
                if (runConfig.particleLengthSigma > 0) {
                    do { // generate random length
                        length = rngPoolPtr->getLN(threadId);
                    } while (length >= maxLength);
                } else {
                    length = runConfig.particleLength;
                }
                double pos[3];
                for (int k = 0; k < 3; k++) {
                    pos[k] = rngPoolPtr->getU01(threadId) * boxEdge[k] + runConfig.initBoxLow[k];
                }
                Equatn orientq;
                getOrient(orientq, runConfig.initOrient[0], runConfig.initOrient[1], runConfig.initOrient[2], threadId);
                double orientation[4];
                Emapq(orientation).coeffs() = orientq.coeffs();
                particleContainer[i] = Particle(i, radius, radius, length, length, pos, orientation);

                // initialize the particle's surface
                // TODO: modify to support multiple particle types
                const Evec3 north = {0.0, 0.0, 1.0};
                sharedPS = SharedParticleSurface("fine", runConfig.particleShapesPtr, north, runConfig.prescribedSurfaceDensityFile);
                particleContainer[i].storeSharedSurface(&sharedPS);
                particleContainer[i].clear();
            }
        }
    }

    if (runConfig.initCircularX) {
        setInitialCircularCrossSection();
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::setInitialCircularCrossSection() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    double radiusCrossSec = 0;            // x, y, z, axis radius
    Evec3 centerCrossSec = Evec3::Zero(); // x, y, z, axis center.
    // x axis
    centerCrossSec = Evec3(0, (runConfig.initBoxHigh[1] - runConfig.initBoxLow[1]) * 0.5 + runConfig.initBoxLow[1],
                           (runConfig.initBoxHigh[2] - runConfig.initBoxLow[2]) * 0.5 + runConfig.initBoxLow[2]);
    radiusCrossSec = 0.5 * std::min(runConfig.initBoxHigh[2] - runConfig.initBoxLow[2],
                                    runConfig.initBoxHigh[1] - runConfig.initBoxLow[1]);
#pragma omp parallel
    {
        const int threadId = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < nLocal; i++) {
            double y = particleContainer[i].pos[1];
            double z = particleContainer[i].pos[2];
            // replace y,z with position in the circle
            getRandPointInCircle(radiusCrossSec, rngPoolPtr->getU01(threadId), rngPoolPtr->getU01(threadId), y, z);
            particleContainer[i].pos[1] = y + centerCrossSec[1];
            particleContainer[i].pos[2] = z + centerCrossSec[2];
        }
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::calcVolFrac() {
    // calc volume fraction of sphero cylinders
    // step 1, calc local total volume
    double volLocal = 0;
    const int nLocal = particleContainer.getNumberOfParticleLocal();
#pragma omp parallel for reduction(+ : volLocal)
    for (int i = 0; i < nLocal; i++) {
        auto &sy = particleContainer[i];
        volLocal += 3.1415926535 * (0.25 * sy.length * pow(sy.radius * 2, 2) + pow(sy.radius * 2, 3) / 6);
    }
    double volGlobal = 0;

    Teuchos::reduceAll(*commRcp, Teuchos::SumValueReductionOp<int, double>(), 1, &volLocal, &volGlobal);

    // step 2, reduce to root and compute total volume
    double boxVolume = (runConfig.simBoxHigh[0] - runConfig.simBoxLow[0]) *
                       (runConfig.simBoxHigh[1] - runConfig.simBoxLow[1]) *
                       (runConfig.simBoxHigh[2] - runConfig.simBoxLow[2]);
    spdlog::warn("Volume Particle = {:g}", volGlobal);
    spdlog::warn("Volume fraction = {:g}", volGlobal / boxVolume);
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::setInitialFromFile(const std::string &filename) {
    spdlog::warn("Reading file " + filename);

    auto parseParticle = [&](Particle &sy, const std::string &line) {
        std::stringstream liness(line);
        // required data
        int gid;
        char type;
        double radius;
        double mx, my, mz;
        double px, py, pz;
        liness >> type >> gid >> radius >> mx >> my >> mz >> px >> py >> pz;
        // optional data
        int group = -1;
        liness >> group;

        Emap3(sy.pos) = Evec3((mx + px), (my + py), (mz + pz)) * 0.5;
        sy.gid = gid;
        sy.group = group;
        sy.isImmovable = (type == 'S') ? true : false;
        sy.radius = radius;
        sy.radiusCollision = radius;
        sy.length = sqrt(pow(px - mx, 2) + pow(py - my, 2) + pow(pz - mz, 2));
        sy.lengthCollision = sy.length;
        if (sy.length > 1e-7) {
            Evec3 direction(px - mx, py - my, pz - mz);
            Emapq(sy.orientation) = Equatn::FromTwoVectors(Evec3(0, 0, 1), direction);
        } else {
            Emapq(sy.orientation) = Equatn::FromTwoVectors(Evec3(0, 0, 1), Evec3(0, 0, 1));
        }
    };

    if (commRcp->getRank() != 0) {
        particleContainer.setNumberOfParticleLocal(0);
    } else {
        std::ifstream myfile(filename);
        std::string line;
        std::getline(myfile, line); // read two header lines
        std::getline(myfile, line);

        std::deque<Particle> particleReadFromFile;
        while (std::getline(myfile, line)) {
            if (line[0] == 'C' || line[0] == 'S') {
                Particle sy;
                parseParticle(sy, line);
                particleReadFromFile.push_back(sy);
            }
        }
        myfile.close();

        spdlog::debug("Particle number in file {} ", particleReadFromFile.size());

        // set on rank 0
        const int nRead = particleReadFromFile.size();
        particleContainer.setNumberOfParticleLocal(nRead);
#pragma omp parallel for
        for (int i = 0; i < nRead; i++) {
            particleContainer[i] = particleReadFromFile[i];
            particleContainer[i].clear();
        }
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::setLinkMapFromFile(const std::string &filename) {
    spdlog::warn("Reading file " + filename);

    auto parseLink = [&](Link &link, const std::string &line) {
        std::stringstream liness(line);
        char header;
        liness >> header >> link.prev >> link.next;
        assert(header == 'L');
    };

    std::ifstream myfile(filename);
    std::string line;
    std::getline(myfile, line); // read two header lines
    std::getline(myfile, line);

    linkMap.clear();
    while (std::getline(myfile, line)) {
        if (line[0] == 'L') {
            Link link;
            parseLink(link, line);
            linkMap.emplace(link.prev, link.next);
            linkReverseMap.emplace(link.next, link.prev);
        }
    }
    myfile.close();

    spdlog::debug("Link number in file {} ", linkMap.size());
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::setInitialFromVTKFile(const std::string &pvtpFileName) {
    spdlog::warn("Reading file " + pvtpFileName);

    if (commRcp->getRank() != 0) {
        particleContainer.setNumberOfParticleLocal(0);
    } else {

        // Read the pvtp file and automatically merge the vtks files into a single polydata
        vtkSmartPointer<vtkXMLPPolyDataReader> reader = vtkSmartPointer<vtkXMLPPolyDataReader>::New();
        reader->SetFileName(pvtpFileName.c_str());
        reader->Update();

        // Extract the polydata (At this point, the polydata is unsorted)
        vtkSmartPointer<vtkPolyData> polydata = reader->GetOutput();
        // geometry data
        vtkSmartPointer<vtkPoints> posData = polydata->GetPoints();
        // Extract the point/cell data
        // int32 types
        vtkSmartPointer<vtkTypeInt32Array> gidData =
            vtkArrayDownCast<vtkTypeInt32Array>(polydata->GetCellData()->GetAbstractArray("gid"));
        vtkSmartPointer<vtkTypeInt32Array> groupData =
            vtkArrayDownCast<vtkTypeInt32Array>(polydata->GetCellData()->GetAbstractArray("group"));
        // unsigned char type
        vtkSmartPointer<vtkTypeUInt8Array> isImmovableData =
            vtkArrayDownCast<vtkTypeUInt8Array>(polydata->GetCellData()->GetAbstractArray("isImmovable"));
        // float/double types
        vtkSmartPointer<vtkDataArray> lengthData = polydata->GetCellData()->GetArray("length");
        vtkSmartPointer<vtkDataArray> lengthCollisionData = polydata->GetCellData()->GetArray("lengthCollision");
        vtkSmartPointer<vtkDataArray> radiusData = polydata->GetCellData()->GetArray("radius");
        vtkSmartPointer<vtkDataArray> radiusCollisionData = polydata->GetCellData()->GetArray("radiusCollision");
        vtkSmartPointer<vtkDataArray> znormData = polydata->GetCellData()->GetArray("znorm");
        vtkSmartPointer<vtkDataArray> velData = polydata->GetCellData()->GetArray("vel");
        vtkSmartPointer<vtkDataArray> omegaData = polydata->GetCellData()->GetArray("omega");

        const int particleNumberInFile = posData->GetNumberOfPoints() / 2; // two points per particle
        particleContainer.setNumberOfParticleLocal(particleNumberInFile);
        spdlog::debug("Particle number in file {} ", particleNumberInFile);

#pragma omp parallel for
        for (int i = 0; i < particleNumberInFile; i++) {
            auto &sy = particleContainer[i];
            double leftEndpointPos[3] = {0, 0, 0};
            double rightEndpointPos[3] = {0, 0, 0};
            posData->GetPoint(i * 2, leftEndpointPos);
            posData->GetPoint(i * 2 + 1, rightEndpointPos);

            Emap3(sy.pos) = (Emap3(leftEndpointPos) + Emap3(rightEndpointPos)) * 0.5;
            sy.gid = gidData->GetComponent(i, 0);
            sy.group = groupData->GetComponent(i, 0);
            sy.isImmovable = isImmovableData->GetTypedComponent(i, 0) > 0 ? true : false;
            sy.length = lengthData->GetComponent(i, 0);
            sy.lengthCollision = lengthCollisionData->GetComponent(i, 0);
            sy.radius = radiusData->GetComponent(i, 0);
            sy.radiusCollision = radiusCollisionData->GetComponent(i, 0);
            const Evec3 direction(znormData->GetComponent(i, 0), znormData->GetComponent(i, 1),
                                  znormData->GetComponent(i, 2));
            Emapq(sy.orientation) = Equatn::FromTwoVectors(Evec3(0, 0, 1), direction);
            sy.vel[0] = velData->GetComponent(i, 0);
            sy.vel[1] = velData->GetComponent(i, 1);
            sy.vel[2] = velData->GetComponent(i, 2);
            sy.omega[0] = omegaData->GetComponent(i, 0);
            sy.omega[1] = omegaData->GetComponent(i, 1);
            sy.omega[2] = omegaData->GetComponent(i, 2);
        }

        // sort the vector of Particles by gid ascending;
        // std::sort(particleReadFromFile.begin(), particleReadFromFile.end(),
        //           [](const Particle &t1, const Particle &t2) { return t1.gid < t2.gid; });
    }
    commRcp->barrier();
}

template <int spectralDegree>
std::string ParticleSystem<spectralDegree>::getCurrentResultFolder() { return getResultFolderWithID(this->snapID); }

template <int spectralDegree>
std::string ParticleSystem<spectralDegree>::getResultFolderWithID(int snapID_) {
    const int num = std::max(400 / commRcp->getSize(), 1); // limit max number of files per folder
    int k = snapID_ / num;
    int low = k * num, high = k * num + num - 1;
    std::string baseFolder =
        "./result/result" + std::to_string(low) + std::string("-") + std::to_string(high) + std::string("/");
    return baseFolder;
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::writeAscii(const std::string &baseFolder) {
    // write a single ascii .dat file
    const int nGlobal = particleContainer.getNumberOfParticleGlobal();

    std::string name = baseFolder + std::string("ParticleAscii_") + std::to_string(snapID) + ".dat";
    ParticleAsciiHeader header;
    header.nparticle = nGlobal;
    header.time = stepCount * runConfig.dt;
    particleContainer.writeParticleAscii(name.c_str(), header);
    if (commRcp->getRank() == 0) {
        FILE *fptr = fopen(name.c_str(), "a");
        for (const auto &key_value : linkMap) {
            fprintf(fptr, "L %d %d\n", key_value.first, key_value.second);
        }
        fclose(fptr);
    }
    commRcp->barrier();
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::writeTimeStepInfo(const std::string &baseFolder) {
    if (commRcp->getRank() == 0) {
        // write a single txt file containing timestep and most recent pvtp file names
        std::string name = baseFolder + std::string("../../TimeStepInfo.txt");
        std::string pvtpFileName = std::string("Particle_") + std::to_string(snapID) + std::string(".pvtp");

        FILE *restartFile = fopen(name.c_str(), "w");
        fprintf(restartFile, "%u\n", restartRngSeed);
        fprintf(restartFile, "%u\n", stepCount);
        fprintf(restartFile, "%u\n", snapID);
        fprintf(restartFile, "%s\n", pvtpFileName.c_str());
        fclose(restartFile);
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::writeVTK(const std::string &baseFolder) {
    const int rank = commRcp->getRank();
    const int size = commRcp->getSize();
    Particle<spectralDegree>::template writeVTP<PS::ParticleSystem<Particle<spectralDegree>>>(
        particleContainer, particleContainer.getNumberOfParticleLocal(),
        baseFolder, std::to_string(snapID), rank);
    conCollectorPtr->writeVTP(baseFolder, "", std::to_string(snapID), rank);
    if (rank == 0) {
        Particle::writePVTP(baseFolder, std::to_string(snapID), size); // write parallel head
        conCollectorPtr->writePVTP(baseFolder, "", std::to_string(snapID), size);
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::writeBox() {
    FILE *boxFile = fopen("./result/simBox.vtk", "w");
    fprintf(boxFile, "# vtk DataFile Version 3.0\n");
    fprintf(boxFile, "vtk file\n");
    fprintf(boxFile, "ASCII\n");
    fprintf(boxFile, "DATASET RECTILINEAR_GRID\n");
    fprintf(boxFile, "DIMENSIONS 2 2 2\n");
    fprintf(boxFile, "X_COORDINATES 2 float\n");
    fprintf(boxFile, "%g %g\n", runConfig.simBoxLow[0], runConfig.simBoxHigh[0]);
    fprintf(boxFile, "Y_COORDINATES 2 float\n");
    fprintf(boxFile, "%g %g\n", runConfig.simBoxLow[1], runConfig.simBoxHigh[1]);
    fprintf(boxFile, "Z_COORDINATES 2 float\n");
    fprintf(boxFile, "%g %g\n", runConfig.simBoxLow[2], runConfig.simBoxHigh[2]);
    fprintf(boxFile, "CELL_DATA 1\n");
    fprintf(boxFile, "POINT_DATA 8\n");
    fclose(boxFile);
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::writeResult() {
    std::string baseFolder = getCurrentResultFolder();
    IOHelper::makeSubFolder(baseFolder);
    writeAscii(baseFolder);
    writeVTK(baseFolder);
    writeTimeStepInfo(baseFolder);
    snapID++;
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::showOnScreenRank0() {
    if (commRcp->getRank() == 0) {
        printf("-----------ParticleSystem Settings-----------\n");
        runConfig.dump();
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::setDomainInfo() {
    const int pbcX = (runConfig.simBoxPBC[0] ? 1 : 0);
    const int pbcY = (runConfig.simBoxPBC[1] ? 1 : 0);
    const int pbcZ = (runConfig.simBoxPBC[2] ? 1 : 0);
    const int pbcFlag = 100 * pbcX + 10 * pbcY + pbcZ;

    switch (pbcFlag) {
    case 0:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_OPEN);
        break;
    case 1:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_Z);
        break;
    case 10:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_Y);
        break;
    case 100:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_X);
        break;
    case 11:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_YZ);
        break;
    case 101:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_XZ);
        break;
    case 110:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_XY);
        break;
    case 111:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_XYZ);
        break;
    }

    PS::F64vec3 rootDomainLow;
    PS::F64vec3 rootDomainHigh;
    for (int k = 0; k < 3; k++) {
        rootDomainLow[k] = runConfig.simBoxLow[k];
        rootDomainHigh[k] = runConfig.simBoxHigh[k];
    }

    dinfo.setPosRootDomain(rootDomainLow, rootDomainHigh); // rootdomain must be specified after PBC
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::decomposeDomain() {
    applyBoxBC();
    dinfo.decomposeDomainAll(particleContainer);
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::exchangeParticle() {
    particleContainer.exchangeParticle(dinfo);
    updateParticleRank();
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::calcMobMatrix() {
    // diagonal hydro mobility operator
    // 3*3 block for translational + 3*3 block for rotational.
    // 3 nnz per row, 18 nnz per tubule

    const double Pi = 3.14159265358979323846;
    const double mu = runConfig.viscosity;

    const int nLocal = particleMapRcp->getNodeNumElements();
    TEUCHOS_ASSERT(nLocal == particleContainer.getNumberOfParticleLocal());
    const int localSize = nLocal * 6; // local row number

    Kokkos::View<size_t *> rowPointers("rowPointers", localSize + 1);
    rowPointers[0] = 0;
    for (int i = 1; i <= localSize; i++) {
        rowPointers[i] = rowPointers[i - 1] + 3;
    }
    Kokkos::View<int *> columnIndices("columnIndices", rowPointers[localSize]);
    Kokkos::View<double *> values("values", rowPointers[localSize]);

#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        const auto &sy = particleContainer[i];

        // calculate the Mob Trans and MobRot
        Emat3 MobTrans; //            double MobTrans[3][3];
        Emat3 MobRot;   //            double MobRot[3][3];
        Emat3 qq;
        Emat3 Imqq;
        Evec3 q = ECmapq(sy.orientation) * Evec3(0, 0, 1);
        qq = q * q.transpose();
        Imqq = Emat3::Identity() - qq;

        double dragPara = 0;
        double dragPerp = 0;
        double dragRot = 0;
        sy.calcDragCoeff(mu, dragPara, dragPerp, dragRot);
        const double dragParaInv = sy.isImmovable ? 0.0 : 1 / dragPara;
        const double dragPerpInv = sy.isImmovable ? 0.0 : 1 / dragPerp;
        const double dragRotInv = sy.isImmovable ? 0.0 : 1 / dragRot;

        MobTrans = dragParaInv * qq + dragPerpInv * Imqq;
        MobRot = dragRotInv * qq + dragRotInv * Imqq; // = dragRotInv * Identity
        // MobRot regularized to remove null space.
        // here it becomes identity matrix,
        // no effect on geometric constraints
        // no problem for axissymetric slender body.
        // this simplifies the rotational Brownian calculations.

        // column index is local index
        columnIndices[18 * i] = 6 * i; // line 1 of Mob Trans
        columnIndices[18 * i + 1] = 6 * i + 1;
        columnIndices[18 * i + 2] = 6 * i + 2;
        columnIndices[18 * i + 3] = 6 * i; // line 2 of Mob Trans
        columnIndices[18 * i + 4] = 6 * i + 1;
        columnIndices[18 * i + 5] = 6 * i + 2;
        columnIndices[18 * i + 6] = 6 * i; // line 3 of Mob Trans
        columnIndices[18 * i + 7] = 6 * i + 1;
        columnIndices[18 * i + 8] = 6 * i + 2;
        columnIndices[18 * i + 9] = 6 * i + 3; // line 1 of Mob Rot
        columnIndices[18 * i + 10] = 6 * i + 4;
        columnIndices[18 * i + 11] = 6 * i + 5;
        columnIndices[18 * i + 12] = 6 * i + 3; // line 2 of Mob Rot
        columnIndices[18 * i + 13] = 6 * i + 4;
        columnIndices[18 * i + 14] = 6 * i + 5;
        columnIndices[18 * i + 15] = 6 * i + 3; // line 3 of Mob Rot
        columnIndices[18 * i + 16] = 6 * i + 4;
        columnIndices[18 * i + 17] = 6 * i + 5;

        values[18 * i] = MobTrans(0, 0); // line 1 of Mob Trans
        values[18 * i + 1] = MobTrans(0, 1);
        values[18 * i + 2] = MobTrans(0, 2);
        values[18 * i + 3] = MobTrans(1, 0); // line 2 of Mob Trans
        values[18 * i + 4] = MobTrans(1, 1);
        values[18 * i + 5] = MobTrans(1, 2);
        values[18 * i + 6] = MobTrans(2, 0); // line 3 of Mob Trans
        values[18 * i + 7] = MobTrans(2, 1);
        values[18 * i + 8] = MobTrans(2, 2);
        values[18 * i + 9] = MobRot(0, 0); // line 1 of Mob Rot
        values[18 * i + 10] = MobRot(0, 1);
        values[18 * i + 11] = MobRot(0, 2);
        values[18 * i + 12] = MobRot(1, 0); // line 2 of Mob Rot
        values[18 * i + 13] = MobRot(1, 1);
        values[18 * i + 14] = MobRot(1, 2);
        values[18 * i + 15] = MobRot(2, 0); // line 3 of Mob Rot
        values[18 * i + 16] = MobRot(2, 1);
        values[18 * i + 17] = MobRot(2, 2);
    }

    // mobMat is block-diagonal, so domainMap=rangeMap
    mobilityMatrixRcp =
        Teuchos::rcp(new TCMAT(particleMobilityMapRcp, particleMobilityMapRcp, rowPointers, columnIndices, values));
    mobilityMatrixRcp->fillComplete(particleMobilityMapRcp, particleMobilityMapRcp); // domainMap, rangeMap

    spdlog::debug("MobMat Constructed " + mobilityMatrixRcp->description());
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::calcMobOperator() {
    calcMobMatrix();
    mobilityOperatorRcp = mobilityMatrixRcp;
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::calcVelocityNonCon() {
    // velocityNonCon = velocityBrown + velocityPartNonBrown + mobility * forcePartNonBrown
    // if monolayer, set velBrownZ =0, velPartNonBrownZ =0, forcePartNonBrownZ =0
    velocityNonConRcp = Teuchos::rcp<TV>(new TV(particleMobilityMapRcp, true)); // allocate and zero out
    auto velNCPtr = velocityNonConRcp->getLocalView<Kokkos::HostSpace>();

    const int nLocal = particleContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(nLocal * 6 == velocityNonConRcp->getLocalLength());

    if (!forcePartNonBrownRcp.is_null()) {
        // apply mobility
        TEUCHOS_ASSERT(!mobilityOperatorRcp.is_null());
        mobilityOperatorRcp->apply(*forcePartNonBrownRcp, *velocityNonConRcp);
        if (runConfig.monolayer) {
#pragma omp parallel for
            for (int i = 0; i < nLocal; i++) {
                velNCPtr(6 * i + 2, 0) = 0; // vz
                velNCPtr(6 * i + 3, 0) = 0; // omegax
                velNCPtr(6 * i + 4, 0) = 0; // omegay
            }
        }
        // write back to Particle members
        auto forcePtr = forcePartNonBrownRcp->getLocalView<Kokkos::HostSpace>();
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            auto &sy = particleContainer[i];
            // torque
            sy.forceNonB[0] = forcePtr(6 * i + 0, 0);
            sy.forceNonB[1] = forcePtr(6 * i + 1, 0);
            sy.forceNonB[2] = forcePtr(6 * i + 2, 0);
            sy.torqueNonB[0] = forcePtr(6 * i + 3, 0);
            sy.torqueNonB[1] = forcePtr(6 * i + 4, 0);
            sy.torqueNonB[2] = forcePtr(6 * i + 5, 0);
        }
    }

    if (!velocityPartNonBrownRcp.is_null()) {
        if (runConfig.monolayer) {
            auto velNBPtr = velocityPartNonBrownRcp->getLocalView<Kokkos::HostSpace>();
#pragma omp parallel for
            for (int i = 0; i < nLocal; i++) {
                velNBPtr(6 * i + 2, 0) = 0; // vz
                velNBPtr(6 * i + 3, 0) = 0; // omegax
                velNBPtr(6 * i + 4, 0) = 0; // omegay
            }
        }
        velocityNonConRcp->update(1.0, *velocityPartNonBrownRcp, 1.0);
    }

    // write back total non Brownian velocity
    // combine and sync the velNonB set in by setForceNonBrown() and setVelocityNonBrown()
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &sy = particleContainer[i];
        // velocity
        sy.velNonB[0] = velNCPtr(6 * i + 0, 0);
        sy.velNonB[1] = velNCPtr(6 * i + 1, 0);
        sy.velNonB[2] = velNCPtr(6 * i + 2, 0);
        sy.omegaNonB[0] = velNCPtr(6 * i + 3, 0);
        sy.omegaNonB[1] = velNCPtr(6 * i + 4, 0);
        sy.omegaNonB[2] = velNCPtr(6 * i + 5, 0);
    }

    // add Brownian motion
    if (!velocityBrownRcp.is_null()) {
        if (runConfig.monolayer) {
            auto velBPtr = velocityBrownRcp->getLocalView<Kokkos::HostSpace>();
#pragma omp parallel for
            for (int i = 0; i < nLocal; i++) {
                velBPtr(6 * i + 2, 0) = 0; // vz
                velBPtr(6 * i + 3, 0) = 0; // omegax
                velBPtr(6 * i + 4, 0) = 0; // omegay
            }
        }
        velocityNonConRcp->update(1.0, *velocityBrownRcp, 1.0);
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::sumForceVelocity() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &sy = particleContainer[i];
        for (int k = 0; k < 3; k++) {
            sy.vel[k] = sy.velNonB[k] + sy.velBrown[k] + sy.velCol[k] + sy.velBi[k];
            sy.omega[k] = sy.omegaNonB[k] + sy.omegaBrown[k] + sy.omegaCol[k] + sy.omegaBi[k];
            sy.force[k] = sy.forceNonB[k] + sy.forceCol[k] + sy.forceBi[k];
            sy.torque[k] = sy.torqueNonB[k] + sy.torqueCol[k] + sy.torqueBi[k];
        }
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::stepEuler() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    const double dt = runConfig.dt;

    if (!runConfig.particleFixed) {
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            auto &sy = particleContainer[i];
            sy.stepEuler(dt);
        }
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::resolveConstraints() {

    Teuchos::RCP<Teuchos::Time> collectColTimer =
        Teuchos::TimeMonitor::getNewCounter("ParticleSystem::CollectCollision");
    Teuchos::RCP<Teuchos::Time> collectLinkTimer = Teuchos::TimeMonitor::getNewCounter("ParticleSystem::CollectLink");

    spdlog::debug("start collect collisions");
    {
        Teuchos::TimeMonitor mon(*collectColTimer);
        collectPairCollision();
        collectBoundaryCollision();
    }

    spdlog::debug("start collect links");
    {
        Teuchos::TimeMonitor mon(*collectLinkTimer);
        collectLinkBilateral();
    }

    // solve collision
    // positive buffer value means collision radius is effectively smaller
    // i.e., less likely to collide
    Teuchos::RCP<Teuchos::Time> solveTimer = Teuchos::TimeMonitor::getNewCounter("ParticleSystem::SolveConstraints");
    {
        Teuchos::TimeMonitor mon(*solveTimer);
        const double buffer = 0;
        spdlog::debug("constraint solver setup");
        conSolverPtr->setup(*conCollectorPtr, mobilityOperatorRcp, velocityNonConRcp, runConfig.dt);
        spdlog::debug("setControl");
        conSolverPtr->setControlParams(runConfig.conResTol, runConfig.conMaxIte, runConfig.conSolverChoice);
        spdlog::debug("solveConstraints");
        conSolverPtr->solveConstraints();
        spdlog::debug("writebackGamma");
        conSolverPtr->writebackGamma();
    }

    saveForceVelocityConstraints();
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::updateParticleMap() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    // setup the new particleMap
    particleMapRcp = getTMAPFromLocalSize(nLocal, commRcp);
    particleMobilityMapRcp = getTMAPFromLocalSize(nLocal * 6, commRcp);

    // setup the globalIndex
    int globalIndexBase = particleMapRcp->getMinGlobalIndex(); // this is a contiguous map
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        particleContainer[i].globalIndex = i + globalIndexBase;
    }
}

template <int spectralDegree>
bool ParticleSystem<spectralDegree>::getIfWriteResultCurrentStep() {
    return (stepCount % static_cast<int>(runConfig.timeSnap / runConfig.dt) == 0);
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::prepareStep() {
    spdlog::warn("CurrentStep {}", stepCount);
    applyBoxBC();

    if (stepCount % 50 == 0) {
        decomposeDomain();
    }

    exchangeParticle();

    const int nLocal = particleContainer.getNumberOfParticleLocal();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &sy = particleContainer[i];
        sy.clear();
        sy.radiusCollision = particleContainer[i].radius * runConfig.particleDiameterColRatio;
        sy.lengthCollision = particleContainer[i].length * runConfig.particleLengthColRatio;
        sy.rank = commRcp->getRank();
        sy.colBuf = runConfig.particleColBuf;
    }

    if (runConfig.monolayer) {
        const double monoZ = (runConfig.simBoxHigh[2] + runConfig.simBoxLow[2]) / 2;
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            auto &sy = particleContainer[i];
            sy.pos[2] = monoZ;
            Evec3 drt = Emapq(sy.orientation) * Evec3(0, 0, 1);
            drt[2] = 0;
            drt.normalize();
            Emapq(sy.orientation).setFromTwoVectors(Evec3(0, 0, 1), drt);
        }
    }

    updateParticleMap();

    buildParticleNearDataDirectory();

    calcMobOperator();

    conCollectorPtr->clear();

    forcePartNonBrownRcp.reset();
    velocityPartNonBrownRcp.reset();
    velocityNonBrownRcp.reset();
    velocityBrownRcp.reset();
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::setForceNonBrown(const std::vector<double> &forceNonBrown) {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(forceNonBrown.size() == 6 * nLocal);
    TEUCHOS_ASSERT(particleMobilityMapRcp->getNodeNumElements() == 6 * nLocal);
    forcePartNonBrownRcp = getTVFromVector(forceNonBrown, commRcp);
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::setVelocityNonBrown(const std::vector<double> &velNonBrown) {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(velNonBrown.size() == 6 * nLocal);
    TEUCHOS_ASSERT(particleMobilityMapRcp->getNodeNumElements() == 6 * nLocal);
    velocityPartNonBrownRcp = getTVFromVector(velNonBrown, commRcp);
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::runStep() {

    if (runConfig.KBT > 0) {
        calcVelocityBrown();
    }

    calcVelocityNonCon();

    resolveConstraints();

    sumForceVelocity();

    if (getIfWriteResultCurrentStep()) {
        // write result before moving. guarantee data written is consistent to geometry
        writeResult();
    }

    stepEuler();

    stepCount++;
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::saveForceVelocityConstraints() {
    // save results
    forceUniRcp = conSolverPtr->getForceUni();
    velocityUniRcp = conSolverPtr->getVelocityUni();
    forceBiRcp = conSolverPtr->getForceBi();
    velocityBiRcp = conSolverPtr->getVelocityBi();

    auto velUniPtr = velocityUniRcp->getLocalView<Kokkos::HostSpace>();
    auto velBiPtr = velocityBiRcp->getLocalView<Kokkos::HostSpace>();
    auto forceUniPtr = forceUniRcp->getLocalView<Kokkos::HostSpace>();
    auto forceBiPtr = forceBiRcp->getLocalView<Kokkos::HostSpace>();

    const int particleLocalNumber = particleContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(velUniPtr.dimension_0() == particleLocalNumber * 6);
    TEUCHOS_ASSERT(velUniPtr.dimension_1() == 1);
    TEUCHOS_ASSERT(velBiPtr.dimension_0() == particleLocalNumber * 6);
    TEUCHOS_ASSERT(velBiPtr.dimension_1() == 1);

#pragma omp parallel for
    for (int i = 0; i < particleLocalNumber; i++) {
        auto &sy = particleContainer[i];
        sy.velCol[0] = velUniPtr(6 * i + 0, 0);
        sy.velCol[1] = velUniPtr(6 * i + 1, 0);
        sy.velCol[2] = velUniPtr(6 * i + 2, 0);
        sy.omegaCol[0] = velUniPtr(6 * i + 3, 0);
        sy.omegaCol[1] = velUniPtr(6 * i + 4, 0);
        sy.omegaCol[2] = velUniPtr(6 * i + 5, 0);
        sy.velBi[0] = velBiPtr(6 * i + 0, 0);
        sy.velBi[1] = velBiPtr(6 * i + 1, 0);
        sy.velBi[2] = velBiPtr(6 * i + 2, 0);
        sy.omegaBi[0] = velBiPtr(6 * i + 3, 0);
        sy.omegaBi[1] = velBiPtr(6 * i + 4, 0);
        sy.omegaBi[2] = velBiPtr(6 * i + 5, 0);

        sy.forceCol[0] = forceUniPtr(6 * i + 0, 0);
        sy.forceCol[1] = forceUniPtr(6 * i + 1, 0);
        sy.forceCol[2] = forceUniPtr(6 * i + 2, 0);
        sy.torqueCol[0] = forceUniPtr(6 * i + 3, 0);
        sy.torqueCol[1] = forceUniPtr(6 * i + 4, 0);
        sy.torqueCol[2] = forceUniPtr(6 * i + 5, 0);
        sy.forceBi[0] = forceBiPtr(6 * i + 0, 0);
        sy.forceBi[1] = forceBiPtr(6 * i + 1, 0);
        sy.forceBi[2] = forceBiPtr(6 * i + 2, 0);
        sy.torqueBi[0] = forceBiPtr(6 * i + 3, 0);
        sy.torqueBi[1] = forceBiPtr(6 * i + 4, 0);
        sy.torqueBi[2] = forceBiPtr(6 * i + 5, 0);
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::calcVelocityBrown() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    const double Pi = 3.1415926535897932384626433;
    const double mu = runConfig.viscosity;
    const double dt = runConfig.dt;
    const double delta = dt * 0.1; // a small parameter used in RFD algorithm
    const double kBT = runConfig.KBT;
    const double kBTfactor = sqrt(2 * kBT / dt);

#pragma omp parallel
    {
        const int threadId = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < nLocal; i++) {
            auto &sy = particleContainer[i];
            // constants
            double dragPara = 0;
            double dragPerp = 0;
            double dragRot = 0;
            sy.calcDragCoeff(mu, dragPara, dragPerp, dragRot);
            const double dragParaInv = sy.isImmovable ? 0.0 : 1 / dragPara;
            const double dragPerpInv = sy.isImmovable ? 0.0 : 1 / dragPerp;
            const double dragRotInv = sy.isImmovable ? 0.0 : 1 / dragRot;

            // convert FDPS vec3 to Evec3
            Evec3 direction = Emapq(sy.orientation) * Evec3(0, 0, 1);

            // RFD from Delong, JCP, 2015
            // slender fiber has 0 rot drag, regularize with identity rot mobility
            // trans mobility is this
            Evec3 q = direction;
            Emat3 Nmat = (dragParaInv - dragPerpInv) * (q * q.transpose()) + (dragPerpInv)*Emat3::Identity();
            Emat3 Nmatsqrt = Nmat.llt().matrixL();

            // velocity
            Evec3 Wrot(rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId));
            Evec3 Wpos(rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId));
            Evec3 Wrfdrot(rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId));
            Evec3 Wrfdpos(rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId));

            Equatn orientRFD = Emapq(sy.orientation);
            EquatnHelper::rotateEquatn(orientRFD, Wrfdrot, delta);
            q = orientRFD * Evec3(0, 0, 1);
            Emat3 Nmatrfd = (dragParaInv - dragPerpInv) * (q * q.transpose()) + (dragPerpInv)*Emat3::Identity();

            Evec3 vel = kBTfactor * (Nmatsqrt * Wpos);           // Gaussian noise
            vel += (kBT / delta) * ((Nmatrfd - Nmat) * Wrfdpos); // rfd drift. seems no effect in this case
            Evec3 omega = sqrt(dragRotInv) * kBTfactor * Wrot;   // regularized identity rotation drag

            Emap3(sy.velBrown) = vel;
            Emap3(sy.omegaBrown) = omega;
        }
    }

    velocityBrownRcp = Teuchos::rcp<TV>(new TV(particleMobilityMapRcp, true));
    auto velocityPtr = velocityBrownRcp->getLocalView<Kokkos::HostSpace>();
    velocityBrownRcp->modify<Kokkos::HostSpace>();

    TEUCHOS_ASSERT(velocityPtr.dimension_0() == nLocal * 6);
    TEUCHOS_ASSERT(velocityPtr.dimension_1() == 1);

#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        const auto &sy = particleContainer[i];
        velocityPtr(6 * i, 0) = sy.velBrown[0];
        velocityPtr(6 * i + 1, 0) = sy.velBrown[1];
        velocityPtr(6 * i + 2, 0) = sy.velBrown[2];
        velocityPtr(6 * i + 3, 0) = sy.omegaBrown[0];
        velocityPtr(6 * i + 4, 0) = sy.omegaBrown[1];
        velocityPtr(6 * i + 5, 0) = sy.omegaBrown[2];
    }
}

template <int spectralDegree>
void CellSystem<spectralDegree>::calcHydroVelocity() {
    PS::ParticleSystem<Sylinder<spectralDegree>> &particleContainer = particleSystem.getContainerNonConst();
    Teuchos::RCP<HydroRodMobility<PS::ParticleSystem<Sylinder<spectralDegree>>>> hydroMobOpRcp =
        Teuchos::rcp(new HydroRodMobility<PS::ParticleSystem<Sylinder<spectralDegree>>, SQWCollector<spectralDegree>>(
            &particleContainer, particleContainer.getNumberOfParticleLocal(), fmmPtr, runConfig,
            particleSystem.runConfig));

    hydroMobOpRcp->calcMotion();
    hydroMobOpRcp->cacheResults(particleContainer);

    const int nLocal = particleContainer.getNumberOfParticleLocal();
    std::vector<double> particleVelOmega(6 * nLocal, 0);

    // hydro & swim velocity
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        const auto &ptc = particleContainer[i];
        // get velocity
        for (int k = 0; k < 3; k++) {
            particleVelOmega[6 * i + k] = ptc.velHydro[k];       // Vel
            particleVelOmega[6 * i + 3 + k] = ptc.omegaHydro[k]; // Omega
        }
    }

    setVelocityNonBrown(particleVelOmega);
}


template <int spectralDegree>
void ParticleSystem<spectralDegree>::collectBoundaryCollision() {
    auto collisionPoolPtr = conCollectorPtr->constraintPoolPtr; // shared_ptr
    const int nThreads = collisionPoolPtr->size();
    const int nLocal = particleContainer.getNumberOfParticleLocal();

    // process collisions with all boundaries
    for (const auto &bPtr : runConfig.boundaryPtr) {
#pragma omp parallel num_threads(nThreads)
        {
            const int threadId = omp_get_thread_num();
            auto &que = (*collisionPoolPtr)[threadId];
#pragma omp for
            for (int i = 0; i < nLocal; i++) {
                const auto &sy = particleContainer[i];
                const Evec3 center = ECmap3(sy.pos);

                // check one point
                auto checkEnd = [&](const Evec3 &Query, const double radius) {
                    double Proj[3], delta[3];
                    bPtr->project(Query.data(), Proj, delta);
                    // if (!bPtr->check(Query.data(), Proj, delta)) {
                    //     printf("boundary projection error\n");
                    // }
                    // if inside boundary, delta = Q-Proj
                    // if outside boundary, delta = Proj-Q
                    double deltanorm = Emap3(delta).norm();
                    Evec3 norm = Emap3(delta) * (1 / deltanorm);
                    Evec3 posI = Query - center;

                    if ((Query - ECmap3(Proj)).dot(ECmap3(delta)) < 0) { // outside boundary
                        que.emplace_back(-deltanorm - radius, 0, sy.gid, sy.gid, sy.globalIndex, sy.globalIndex,
                                         norm.data(), norm.data(), posI.data(), posI.data(), Query.data(), Proj, true,
                                         false, 0.0, 0.0);
                    } else if (deltanorm <
                               (1 + runConfig.particleColBuf * 2) * sy.radiusCollision) { // inside boundary but close
                        que.emplace_back(deltanorm - radius, 0, sy.gid, sy.gid, sy.globalIndex, sy.globalIndex,
                                         norm.data(), norm.data(), posI.data(), posI.data(), Query.data(), Proj, true,
                                         false, 0.0, 0.0);
                    }
                };

                if (sy.isSphere(true)) {
                    double radius = sy.lengthCollision * 0.5 + sy.radiusCollision;
                    checkEnd(center, radius);
                } else {
                    const Equatn orientation = ECmapq(sy.orientation);
                    const Evec3 direction = orientation * Evec3(0, 0, 1);
                    const double length = sy.lengthCollision;
                    const Evec3 Qm = center - direction * (length * 0.5);
                    const Evec3 Qp = center + direction * (length * 0.5);
                    checkEnd(Qm, sy.radiusCollision);
                    checkEnd(Qp, sy.radiusCollision);
                }
            }
        }
    }
    return;
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::collectPairCollision() {

    CalcParticleNearForce calcColFtr(conCollectorPtr->constraintPoolPtr);

    TEUCHOS_ASSERT(treeParticleNearPtr);
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    setTreeParticle();
    treeParticleNearPtr->calcForceAll(calcColFtr, particleContainer, dinfo);
}

template <int spectralDegree>
std::pair<int, int> ParticleSystem<spectralDegree>::getMaxGid() {
    int maxGidLocal = 0;
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    for (int i = 0; i < nLocal; i++) {
        maxGidLocal = std::max(maxGidLocal, particleContainer[i].gid);
    }

    int maxGidGlobal = maxGidLocal;
    Teuchos::reduceAll(*commRcp, Teuchos::MaxValueReductionOp<int, int>(), 1, &maxGidLocal, &maxGidGlobal);
    spdlog::warn("rank: {}, maxGidLocal: {}, maxGidGlobal {}", commRcp->getRank(), maxGidLocal, maxGidGlobal);

    return std::pair<int, int>(maxGidLocal, maxGidGlobal);
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::calcBoundingBox(double localLow[3], double localHigh[3], double globalLow[3],
                                     double globalHigh[3]) {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    double lx, ly, lz;
    lx = ly = lz = std::numeric_limits<double>::max();
    double hx, hy, hz;
    hx = hy = hz = std::numeric_limits<double>::min();

    for (int i = 0; i < nLocal; i++) {
        const auto &sy = particleContainer[i];
        const Evec3 direction = ECmapq(sy.orientation) * Evec3(0, 0, 1);
        Evec3 pm = ECmap3(sy.pos) - (sy.length * 0.5) * direction;
        Evec3 pp = ECmap3(sy.pos) + (sy.length * 0.5) * direction;
        lx = std::min(std::min(lx, pm[0]), pp[0]);
        ly = std::min(std::min(ly, pm[1]), pp[1]);
        lz = std::min(std::min(lz, pm[2]), pp[2]);
        hx = std::max(std::max(hx, pm[0]), pp[0]);
        hy = std::max(std::max(hy, pm[1]), pp[1]);
        hz = std::max(std::max(hz, pm[2]), pp[2]);
    }

    localLow[0] = lx;
    localLow[1] = ly;
    localLow[2] = lz;
    localHigh[0] = hx;
    localHigh[1] = hy;
    localHigh[2] = hz;

    for (int k = 0; k < 3; k++) {
        globalLow[k] = localLow[k];
        globalHigh[k] = localHigh[k];
    }

    Teuchos::reduceAll(*commRcp, Teuchos::MinValueReductionOp<int, double>(), 3, localLow, globalLow);
    Teuchos::reduceAll(*commRcp, Teuchos::MaxValueReductionOp<int, double>(), 3, localHigh, globalHigh);

    return;
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::updateParticleRank() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    const int rank = commRcp->getRank();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        particleContainer[i].rank = rank;
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::applyBoxBC() { particleContainer.adjustPositionIntoRootDomain(dinfo); }

template <int spectralDegree>
void ParticleSystem<spectralDegree>::calcConStress() {
    if (runConfig.logLevel > spdlog::level::info)
        return;

    Emat3 sumBiStress = Emat3::Zero();
    Emat3 sumUniStress = Emat3::Zero();
    conCollectorPtr->sumLocalConstraintStress(sumUniStress, sumBiStress, false);

    // scale to nkBT
    const double scaleFactor = 1 / (particleMapRcp->getGlobalNumElements() * runConfig.KBT);
    sumBiStress *= scaleFactor;
    sumUniStress *= scaleFactor;
    // mpi reduction
    double uniStressLocal[9];
    double biStressLocal[9];
    double uniStressGlobal[9];
    double biStressGlobal[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            uniStressLocal[i * 3 + j] = sumUniStress(i, j);
            uniStressGlobal[i * 3 + j] = 0;
            biStressLocal[i * 3 + j] = sumBiStress(i, j);
            biStressGlobal[i * 3 + j] = 0;
        }
    }

    Teuchos::reduceAll(*commRcp, Teuchos::SumValueReductionOp<int, double>(), 9, uniStressLocal, uniStressGlobal);
    Teuchos::reduceAll(*commRcp, Teuchos::SumValueReductionOp<int, double>(), 9, biStressLocal, biStressGlobal);

    spdlog::info("RECORD: ColXF,{:g},{:g},{:g},{:g},{:g},{:g},{:g},{:g},{:g}", //
                 uniStressGlobal[0], uniStressGlobal[1], uniStressGlobal[2],   //
                 uniStressGlobal[3], uniStressGlobal[4], uniStressGlobal[5],   //
                 uniStressGlobal[6], uniStressGlobal[7], uniStressGlobal[8]);
    spdlog::info("RECORD: BiXF,{:g},{:g},{:g},{:g},{:g},{:g},{:g},{:g},{:g}", //
                 biStressGlobal[0], biStressGlobal[1], biStressGlobal[2],     //
                 biStressGlobal[3], biStressGlobal[4], biStressGlobal[5],     //
                 biStressGlobal[6], biStressGlobal[7], biStressGlobal[8]);
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::calcOrderParameter() {
    if (runConfig.logLevel > spdlog::level::info)
        return;

    double px = 0, py = 0, pz = 0;    // pvec
    double Qxx = 0, Qxy = 0, Qxz = 0; // Qtensor
    double Qyx = 0, Qyy = 0, Qyz = 0; // Qtensor
    double Qzx = 0, Qzy = 0, Qzz = 0; // Qtensor

    const int nLocal = particleContainer.getNumberOfParticleLocal();

#pragma omp parallel for reduction(+ : px, py, pz, Qxx, Qxy, Qxz, Qyx, Qyy, Qyz, Qzx, Qzy, Qzz)
    for (int i = 0; i < nLocal; i++) {
        const auto &sy = particleContainer[i];
        const Evec3 direction = ECmapq(sy.orientation) * Evec3(0, 0, 1);
        px += direction.x();
        py += direction.y();
        pz += direction.z();
        const Emat3 Q = direction * direction.transpose() - Emat3::Identity() * (1 / 3.0);
        Qxx += Q(0, 0);
        Qxy += Q(0, 1);
        Qxz += Q(0, 2);
        Qyx += Q(1, 0);
        Qyy += Q(1, 1);
        Qyz += Q(1, 2);
        Qzx += Q(2, 0);
        Qzy += Q(2, 1);
        Qzz += Q(2, 2);
    }

    // global average
    const int nGlobal = particleContainer.getNumberOfParticleGlobal();
    double pQ[12] = {px, py, pz, Qxx, Qxy, Qxz, Qyx, Qyy, Qyz, Qzx, Qzy, Qzz};
    MPI_Allreduce(MPI_IN_PLACE, pQ, 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < 12; i++) {
        pQ[i] *= (1.0 / nGlobal);
    }

    spdlog::info("RECORD: Order P,{:g},{:g},{:g},Q,{:g},{:g},{:g},{:g},{:g},{:g},{:g},{:g},{:g}", //
                 pQ[0], pQ[1], pQ[2],                                                             // pvec
                 pQ[3], pQ[4], pQ[5],                                                             // Qtensor
                 pQ[6], pQ[7], pQ[8],                                                             // Qtensor
                 pQ[9], pQ[10], pQ[11]                                                            // Qtensor
    );
}

template <int spectralDegree>
std::vector<int> ParticleSystem<spectralDegree>::addNewParticle(const std::vector<Particle> &newParticle) {
    // assign unique new gid for particles on all ranks
    std::pair<int, int> maxGid = getMaxGid();
    const int maxGidLocal = maxGid.first;
    const int maxGidGlobal = maxGid.second;
    const int newCountLocal = newParticle.size();

    // collect number of ids from all ranks to rank0
    std::vector<int> newCount(commRcp->getSize(), 0);
    std::vector<int> displ(commRcp->getSize() + 1, 0);
    MPI_Gather(&newCountLocal, 1, MPI_INT, newCount.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> newGid;
    if (commRcp->getRank() == 0) {
        // generate random gid on rank 0
        std::partial_sum(newCount.cbegin(), newCount.cend(), displ.begin() + 1);
        const int newCountGlobal = displ.back();
        newGid.resize(newCountGlobal, 0);
        std::iota(newGid.begin(), newGid.end(), maxGidGlobal + 1);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(newGid.begin(), newGid.end(), g);
    } else {
        newGid.resize(newCountLocal, 0);
    }

    // scatter from rank 0 to every rank
    std::vector<int> newGidRecv(newCountLocal, 0);
    MPI_Scatterv(newGid.data(), newCount.data(), displ.data(), MPI_INT, //
                 newGidRecv.data(), newCountLocal, MPI_INT, 0, MPI_COMM_WORLD);

    // set new gid
    for (int i = 0; i < newCountLocal; i++) {
        Particle sy = newParticle[i];
        sy.gid = newGidRecv[i];
        particleContainer.addOneParticle(sy);
    }

    return newGidRecv;
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::addNewLink(const std::vector<Link> &newLink) {
    // synchronize newLink to all mpi ranks
    const int newCountLocal = newLink.size();
    std::vector<int> newCount(commRcp->getSize(), 0);
    MPI_Allgather(&newCountLocal, 1, MPI_INT, newCount.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> displ(commRcp->getSize() + 1, 0);
    std::partial_sum(newCount.cbegin(), newCount.cend(), displ.begin() + 1);
    std::vector<Link> newLinkRecv(displ.back());
    MPI_Allgatherv(newLink.data(), newCountLocal, createMPIStructType<Link>(), newLinkRecv.data(), newCount.data(),
                   displ.data(), createMPIStructType<Link>(), MPI_COMM_WORLD);

    // put newLinks into the map, same op on all mpi ranks
    for (const auto &ll : newLinkRecv) {
        linkMap.emplace(ll.prev, ll.next);
        linkReverseMap.emplace(ll.next, ll.prev);
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::buildParticleNearDataDirectory() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    auto &particleNearDataDirectory = *particleNearDataDirectoryPtr;
    particleNearDataDirectory.gidOnLocal.resize(nLocal);
    particleNearDataDirectory.dataOnLocal.resize(nLocal);
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        particleNearDataDirectory.gidOnLocal[i] = particleContainer[i].gid;
        particleNearDataDirectory.dataOnLocal[i].copyFromFP(particleContainer[i]);
    }

    // build index
    particleNearDataDirectory.buildIndex();
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::collectLinkBilateral() {
    // setup bilateral link constraints
    // need special treatment of periodic boundary conditions

    const int nLocal = particleContainer.getNumberOfParticleLocal();
    auto &conPool = *(this->conCollectorPtr->constraintPoolPtr);
    if (conPool.size() != omp_get_max_threads()) {
        spdlog::critical("conPool multithread mismatch error");
        std::exit(1);
    }

    // fill the data to find
    auto &gidToFind = particleNearDataDirectoryPtr->gidToFind;
    const auto &dataToFind = particleNearDataDirectoryPtr->dataToFind;

    std::vector<int> gidDisp(nLocal + 1, 0);
    gidToFind.clear();
    gidToFind.reserve(nLocal);

    // loop over all particles
    // if linkMap[sy.gid] not empty, find info for all next
    for (int i = 0; i < nLocal; i++) {
        const auto &sy = particleContainer[i];
        const auto &range = linkMap.equal_range(sy.gid);
        int count = 0;
        for (auto it = range.first; it != range.second; it++) {
            gidToFind.push_back(it->second); // next
            count++;
        }
        gidDisp[i + 1] = gidDisp[i] + count; // number of links for each local Particle
    }

    particleNearDataDirectoryPtr->find();

#pragma omp parallel
    {
        const int threadId = omp_get_thread_num();
        auto &conQue = conPool[threadId];
#pragma omp for
        for (int i = 0; i < nLocal; i++) {
            const auto &syI = particleContainer[i]; // particle
            const int lb = gidDisp[i];
            const int ub = gidDisp[i + 1];

            for (int j = lb; j < ub; j++) {
                const auto &syJ = particleNearDataDirectoryPtr->dataToFind[j]; // particleNear

                const Evec3 &centerI = ECmap3(syI.pos);
                Evec3 centerJ = ECmap3(syJ.pos);
                // apply PBC on centerJ
                for (int k = 0; k < 3; k++) {
                    if (!runConfig.simBoxPBC[k])
                        continue;
                    double trg = centerI[k];
                    double xk = centerJ[k];
                    findPBCImage(runConfig.simBoxLow[k], runConfig.simBoxHigh[k], xk, trg);
                    centerJ[k] = xk;
                    // error check
                    if (fabs(trg - xk) > 0.5 * (runConfig.simBoxHigh[k] - runConfig.simBoxLow[k])) {
                        spdlog::critical("pbc image error in bilateral links");
                        std::exit(1);
                    }
                }
                // particles are not treated as spheres for bilateral constraints
                // constraint is always added between Pp and Qm
                // constraint target length is radiusI + radiusJ + runConfig.linkGap
                const Evec3 directionI = ECmapq(syI.orientation) * Evec3(0, 0, 1);
                const Evec3 Pp = centerI + directionI * (0.5 * syI.length); // plus end
                const Evec3 directionJ = ECmap3(syJ.direction);
                const Evec3 Qm = centerJ - directionJ * (0.5 * syJ.length);
                const Evec3 Ploc = Pp;
                const Evec3 Qloc = Qm;
                const Evec3 rvec = Qloc - Ploc;
                const double rnorm = rvec.norm();
                const double delta0 = rnorm - syI.radius - syJ.radius - runConfig.linkGap;
                const double gamma = delta0 < 0 ? -delta0 : 0;
                const Evec3 normI = (Ploc - Qloc).normalized();
                const Evec3 normJ = -normI;
                const Evec3 posI = Ploc - centerI;
                const Evec3 posJ = Qloc - centerJ;
                ConstraintBlock conBlock(delta0, gamma,              // current separation, initial guess of gamma
                                         syI.gid, syJ.gid,           //
                                         syI.globalIndex,            //
                                         syJ.globalIndex,            //
                                         normI.data(), normJ.data(), // direction of collision force
                                         posI.data(), posJ.data(), // location of collision relative to particle center
                                         Ploc.data(), Qloc.data(), // location of collision in lab frame
                                         false, true, runConfig.linkKappa, 0.0);
                Emat3 stressIJ;
                CalcParticleNearForce::collideStress(directionI, directionJ, centerI, centerJ, syI.length, syJ.length,
                                                     syI.radius, syJ.radius, 1.0, Ploc, Qloc, stressIJ);
                conBlock.setStress(stressIJ);
                conQue.push_back(conBlock);
            }
        }
    }
}

template <int spectralDegree>
void ParticleSystem<spectralDegree>::printTimingSummary(const bool zeroOut) {
    if (runConfig.timerLevel <= spdlog::level::info)
        Teuchos::TimeMonitor::summarize();
    if (zeroOut)
        Teuchos::TimeMonitor::zeroOutTimers();
}
