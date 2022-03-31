#include "HydroParticleOperator.hpp"
#include "Particle/ParticleConfig.hpp"
#include "Trilinos/Preconditioner.hpp"
#include "Trilinos/TpetraUtil.hpp"
#include "Util/EigenDef.hpp"
#include <boost/math/special_functions/legendre.hpp>

// TODO
//  1. (done) Start by assuming that everything is double but then switch from double to std::complex<double> where
//   necessary
//  2. (done) Switch surface functions to be primarily void
//  3. (done) Replace np.cross with something else!
//  4. (done) All functions should use and alpha/beta to allow for Y := alpha Ax + beta Y
//  5. (done) getUserInputSurfaceDensityCoeff needs updated to allow for data read from csv
//  6. (done) Figure out the syntax for hyperinterpolation and for rotating the quadrature
//  7.

template <class Container>
HydroParticleOperator<Container>::HydroParticleOperator(const Container *const particleContainerPtr,
                                                        const int numPtcLocal, const ParticleConfig &runConfig)
    : particleContainerPtr(particleContainerPtr), numPtcLocal(numPtcLocal), runConfig(runConfig) {
    commRcp = getMPIWORLDTCOMM();
    setupDOF();
}

template <class Container>
void HydroParticleOperator<Container>::setupDOF() {
    // task: initialize all data structures independent of fe data
    particlePts.resize(numPtcLocal);
    particlePtsIndex.resize(numPtcLocal + 1, 0);
    const auto &particleContainer = *particleContainerPtr;
    for (int i = 0; i < numPtcLocal; i++) {
        particlePts[i] = particleContainer[i].numQuadPt;
        particlePtsIndex[i + 1] = particlePtsIndex[i] + particlePts[i];
    }

    particleMapRcp = getTMAPFromLocalSize(numPtcLocal, commRcp);
    particlePtsMapRcp = getTMAPFromLocalSize(particlePtsIndex.back(), commRcp);
    pointValuesMapRcp = getTMAPFromLocalSize(3 * particlePtsIndex.back(), commRcp);

    const int nQuadLocal = particlePtsIndex.back();

    // preallocate spaces
    FTExt.resize(6 * numPtcLocal, 0); ///< external force/torque specified in HydroTable, 6 dof per particle
    Hu.resize(numPtcLocal, 0);        ///< <us(s),1> integral, 1 dof per particle
    Hf.resize(numPtcLocal, 0);        ///< <fs(s),1> integral, 1 dof per particle

    // these two are original [-1,1] data, not scaled by actual length
    sloc.resize(nQuadLocal, 0);         ///< Q dof per particle, quadrature points in ([-1,1])
    weight.resize(nQuadLocal, 0);       ///< Q dof per particle, quadrature weights
    slipVelocity.resize(nQuadLocal, 0); ///< Q dof per particle, slipVelocity for all particles
    activeStress.resize(nQuadLocal, 0); ///< Q dof per particle, activeStress for all particles
    radius.resize(nQuadLocal, 0);       ///< Q dof per particle, radius for all particles
    fsRcp = Teuchos::rcp(new TV(pointValuesMapRcp, true));
    auto fsPtr = fsRcp->getLocalView<Kokkos::HostSpace>();
    fhRcp = Teuchos::rcp(new TV(pointValuesMapRcp, true));
    auto fhPtr = fhRcp->getLocalView<Kokkos::HostSpace>();

    // temporary data for uinf and their integrals
    uinf.resize(3 * nQuadLocal, 0); ///< 3Q dof per particle,
    Oinf.resize(3 * nQuadLocal, 0); ///< 3 dof per particle,
    Sinf.resize(3 * nQuadLocal, 0); ///< 3 dof per particle,

#pragma omp parallel for
    for (int i = 0; i < numPtcLocal; i++) {
        const auto &sy = particleContainer[i];
        const double length = sy.length;
        const int numQuadPt = sy.numQuadPt;
        const auto quadPtr = sy.quadPtr;
        assert(numQuadPt == particlePts[i]);
        assert(sy.numQuadPt == quadPtr->getSize());

        const Evec3 direction = ECmapq(sy.orientation) * Evec3(0, 0, 1);
        const double *sQuadPt = quadPtr->getPoints();
        const double *weightQuadPt = quadPtr->getWeights();

        std::vector<double> slipVelocityPt(numQuadPt, 0);
        std::vector<double> activeStressPt(numQuadPt, 0);
        std::vector<double> radiusPt(numQuadPt, 0);
        // setup data for quadrature points according to cell species
        const Hydro *pHydro = &(cellConfig.cellHydro.at(sy.speciesID));
        const auto &table = pHydro->cellHydroTable;
        table.getValue(numQuadPt, quadPtr->getPoints(), slipVelocityPt.data(), activeStressPt.data(), radiusPt.data());

        Hu[i] = length * 0.5 * quadPtr->intSamples(slipVelocityPt.data());
        Hf[i] = length * 0.5 * quadPtr->intSamples(activeStressPt.data());
        FTExt[6 * i + 0] = pHydro->forceExternal[0];
        FTExt[6 * i + 1] = pHydro->forceExternal[1];
        FTExt[6 * i + 2] = pHydro->forceExternal[2];
        FTExt[6 * i + 3] = pHydro->torqueExternal[0];
        FTExt[6 * i + 4] = pHydro->torqueExternal[1];
        FTExt[6 * i + 5] = pHydro->torqueExternal[2];

        const int idx = particlePtsIndex[i];
        for (int j = 0; j < numQuadPt; j++) {
            sloc[idx + j] = sQuadPt[j];
            weight[idx + j] = weightQuadPt[j];
            slipVelocity[idx + j] = slipVelocityPt[j];
            activeStress[idx + j] = activeStressPt[j];
            radius[idx + j] = radiusPt[j];
            fsPtr(3 * (idx + j) + 0, 0) = activeStressPt[j] * direction[0];
            fsPtr(3 * (idx + j) + 1, 0) = activeStressPt[j] * direction[1];
            fsPtr(3 * (idx + j) + 2, 0) = activeStressPt[j] * direction[2];
            fhPtr(3 * (idx + j) + 0, 0) = sy.forceHydro[3 * j + 0];
            fhPtr(3 * (idx + j) + 1, 0) = sy.forceHydro[3 * j + 1];
            fhPtr(3 * (idx + j) + 2, 0) = sy.forceHydro[3 * j + 2];
        }
    }
}

template <class Container>
void HydroParticleOperator<Container>::cacheResults(Container &particleContainer) const {
#pragma omp parallel for
    for (int pidx = 0; pidx < numPtcLocal; pidx++) {
        // particle properties
        const auto &ptc = particleContainer[pidx];
        const auto surface = ptc.getSurface();
        const int numGridPts = surface.getNumGridPts();

        // cache the particle velocity/omega
        ptc.velHydro[0] += particleVelOmega[6 * pidx + 0];
        ptc.velHydro[1] += particleVelOmega[6 * pidx + 1];
        ptc.velHydro[2] += particleVelOmega[6 * pidx + 2];
        ptc.omegaHydro[0] += particleVelOmega[6 * pidx + 3];
        ptc.omegaHydro[1] += particleVelOmega[6 * pidx + 4];
        ptc.omegaHydro[2] += particleVelOmega[6 * pidx + 5];

        // cache surface density/velocity values
        const int pointidx = self.particlePtsIndex[pidx];
        const int coeffidx = self.particleCoeffIndex[pidx];

        for (int i = 0; i < numGridPts; i++) {
            const int idxPoint = 3 * (coeffidx + i);
            layer.surfaceDensityGridVals[i, 0] =
                knownSurfaceDensityGridVals[idxPoint + 0] + unknownSurfaceDensityGridVals[idxPoint + 0];
            layer.surfaceDensityGridVals[i, 1] =
                knownSurfaceDensityGridVals[idxPoint + 1] + unknownSurfaceDensityGridVals[idxPoint + 1];
            layer.surfaceDensityGridVals[i, 2] =
                knownSurfaceDensityGridVals[idxPoint + 2] + unknownSurfaceDensityGridVals[idxPoint + 2];
            layer.surfaceVelocityGridVals[i, 0] =
                knownSurfaceVelocityGridVals[idxPoint + 0] + unknownSurfaceVelocityGridVals[idxPoint + 0];
            layer.surfaceVelocityGridVals[i, 1] =
                knownSurfaceVelocityGridVals[idxPoint + 1] + unknownSurfaceVelocityGridVals[idxPoint + 1];
            layer.surfaceVelocityGridVals[i, 2] =
                knownSurfaceVelocityGridVals[idxPoint + 2] + unknownSurfaceVelocityGridVals[idxPoint + 2];
                
            const int idxCoeff = 3 * (coeffidx + i);
            layer.surfaceDensityCoeff[i, 0] =
                knownSurfaceDensityCoeff[idxCoeff + 0] + unknownSurfaceDensityCoeff[idxCoeff + 0];
            layer.surfaceDensityCoeff[i, 1] =
                knownSurfaceDensityCoeff[idxCoeff + 1] + unknownSurfaceDensityCoeff[idxCoeff + 1];
            layer.surfaceDensityCoeff[i, 2] =
                knownSurfaceDensityCoeff[idxCoeff + 2] + unknownSurfaceDensityCoeff[idxCoeff + 2];
            layer.surfaceVelocityCoeff[i, 0] =
                knownSurfaceVelocityCoeff[idxCoeff + 0] + unknownSurfaceVelocityCoeff[idxCoeff + 0];
            layer.surfaceVelocityCoeff[i, 1] =
                knownSurfaceVelocityCoeff[idxCoeff + 1] + unknownSurfaceVelocityCoeff[idxCoeff + 1];
            layer.surfaceVelocityCoeff[i, 2] =
                knownSurfaceVelocityCoeff[idxCoeff + 2] + unknownSurfaceVelocityCoeff[idxCoeff + 2];
        }
    }
}

template <class Container>
void HydroParticleOperator<Container>::getSurfaceDensityCoeffFromParticleContainer(
    std::complex<double> *surfaceDensityCoeff, const double alpha = 1.0, const double beta = 0.0) const {
    const auto &particleContainer = *particleContainerPtr;

#pragma omp parallel for
    for (int pidx = 0; pidx < numPtcLocal; pidx++) {
        // particle properties
        const auto &ptc = particleContainer[pidx];
        const auto surface = ptc.getSurface();
        const int numGridPts = surface.getNumGridPts();

        // reconstruct surfaceDensity from surfaceDensityCoeff
        const int pointidx = self.particlePtsIndex[pidx];
        const int coeffidx = self.particleCoeffIndex[pidx];

        for (int i = 0; i < numGridPts; i++) {
            surfaceDensityCoeff[3 * (coeffidx + i) + 0] =
                beta * surfaceDensityCoeff[3 * (coeffidx + i) + 0] + alpha * layer.surfaceDensityCoeff[3 * i + 0];
            surfaceDensityCoeff[3 * (coeffidx + i) + 1] =
                beta * surfaceDensityCoeff[3 * (coeffidx + i) + 1] + alpha * layer.surfaceDensityCoeff[3 * i + 1];
            surfaceDensityCoeff[3 * (coeffidx + i) + 2] =
                beta * surfaceDensityCoeff[3 * (coeffidx + i) + 2] + alpha * layer.surfaceDensityCoeff[3 * i + 2];
        }
    }
}

template <class Container>
void HydroParticleOperator<Container>::getUserInputSurfaceDensityCoeff(
    std::complex<double> *userInputSurfaceDensityCoeff, const double alpha = 1.0, const double beta = 0.0) const {
    const auto &particleContainer = *particleContainerPtr;

    // step 1: get the surface density induced by the user input force and torque
    measureBx(totalExtForceTorque.data(), userInputSurfaceDensityCoeff, alpha, beta);

    // step 2: get the surface density coefficients from the user input csv
#pragma omp parallel for
    for (int pidx = 0; pidx < numPtcLocal; pidx++) {
        // particle properties
        const auto &ptc = particleContainer[pidx];
        const auto surface = ptc.getSurface();
        const int numSpectralCoeff = surface.getNumSpectralCoeff();

        // reconstruct surfaceDensity from surfaceDensityCoeff
        const int coeffidx = self.particleCoeffIndex[pidx];

        // get and store the prescribed coeffs
        std::complex<double> *userInputSurfaceDensityCoeffi = *(userInputSurfaceDensityCoeff + 3 * coeffidx);
        surface.getUserInputSurfaceDensity(userInputSurfaceDensityCoeffi, alpha, 1.0);
    }
}

template <class Container>
void HydroParticleOperator<Container>::calcParticleVelOmega() const {
    // center of mass velocity and omega of each particle enduced by the known/unknown surfaceDensity
    const auto &particleContainer = *particleContainerPtr;

#pragma omp parallel for
    for (int pidxTarget = 0; pidxTarget < numPtcLocal; pidxTarget++) {
        // target particle properties
        const auto &ptcTarget = particleContainer[pidxTarget];
        const auto surfaceTarget = ptcTarget.getSurface();
        const int numGridPtsTarget = surfaceTarget.getNumGridPts();
        const int numSpectralCoeffTarget = surfaceTarget.getNumSpectralCoeff();
        const Evec3 &centroidTarget = ptcTarget.pos;
        const Emat3 invMomentOfInertiaTensorTarget = surfaceTarget.getInvMomentOfInertiaTensorCurrentConfig();
        const double invSurfaceAreaTarget = 1.0 / surfaceTarget.getSurfaceArea();

        // solve for surface velocity at each point
        Evec3 vel(0.0, 0.0, 0.0);
        Evec3 omega(0.0, 0.0, 0.0);
        std::vector<double> surfaceVelGrid(3 * numGridPtsTarget);
        for (int i = 0; i < numGridPtsTarget; i++) {
            const Evec3 gridPointTargeti = surfaceTarget.getGridPointCurrentConfig(ptcTarget.pos, i);
            const Evec3 spherePointTargeti = surfaceTarget.getSpherePointRefConfig(i);

            for (int pidxSource = 0; pidxSource < numPtcLocal; pidxSource++) {
                // source particle properties
                const auto &ptcSource = particleContainer[pidxSource];
                const auto surfaceSource = ptcSource.getSurface();
                const int numGridPtsSource = surfaceSource.getNumGridPts();
                const int ptsidxSource = self.particlePtsIndex[pidxSource];

                if (pidxSource == pidxTarget) {
                    // particle - self interaction

                    // step 1. Setup quadrature with the target point as "north"
                    const sharedPS = SharedParticleSurface("fineRotated", runConfig.particleShapesPtr, spherePointTargeti);

                    // step 2. Use hyperinterpolation to evaluate the grid quantities
                    //           at the rotated quadrature points
                    const int coeffidx = self.particleCoeffIndex[pidxSource];

                    const std::complex<double> *coeffVSHPtr = *(surfaceDensityCoeff + 3 * coeffidx);
                    std::vector<double> surfaceDensityHyperinterpolated(3 * numGridPtsSource);
                    surface.reconstructSurfaceVectorFcn(coeffVSHPtr, surfaceDensityHyperinterpolated.data());

                    // step 3. Apply the addition theorem to the modified Stokeslet
                    for (int j = 0; j < numGridPtsSource; j++) {
                        const Evec3 xHat = spherePointTargeti;
                        const Evec3 yHat = spherePointsSourceRotated [j, :];
                        const Evec3 px = gridPointTargeti;
                        const Evec3 py = gridPointsSourceRotated [j, :];

                        const Evev3 rVec = yHat - xHat;
                        const Evev3 prVec = py - px;

                        if (rVec.norm() < 1e-8) {
                            // when r == 0, the weight is zero and skipping is allowed
                            // this if statement will never be entered for endpoint=False
                            continue;
                        }

                        const double rRatio = rVec.norm() / prVec.norm();
                        const Evec3 mPhi = rRatio * calcRSrokeslet(prVec) * ECmap3(surfaceDensityHyperinterpolated[j]);

                        double alpha = 0.0;
                        for (int l = 0; l <= surfaceSource.order; l++) {
                            alpha += legendre_p(l, xHat.dot(yHat));
                        }
                        Emap3(surfaceVelGrid.data() + 3 * i) += alpha * mPhi * surfaceWeightsSource[j];
                    }
                } else {
                    // particle - particle interaction
                    for (int j = 0; j < numGridPtsSource; j++) {
                        const Evec3 gridPointSourcej = surfaceSource.getGridPointCurrentConfig(ptcSource.pos, j);
                        const Evec3 rVec = gridPointTargeti - gridPointSourcej;
                        const Evec3 surfaceDensitySourcej = ECmap3(surfaceDensityGridValsPtr + 3 * (ptsidxSource + j));

                        const Evec3 velocityVec = calcStokeslet(rVec, surfaceDensitySourcej);
                        Emap3(surfaceVelGrid.data() + 3 * i) += velocityVec * surfaceSource.getGridWeight(j);
                    }
                }
            }

            // calculate and store vel/omega at ptcTarget.pos
            const Evec3 surfaceVelGridi = ECmap3(surfaceVelGrid.data() + 3 * i);
            vel += surfaceVelGridi * surfaceTarget.getGridWeight(i);
            omega += (gridPointTargeti - centroidTarget).cross(surfaceVelGridi) * surfaceTarget.getGridWeight(i);
        }

        // store vel/omega
        Emap3(particleVelOmega.data() + 6 * pidxTarget) = vel * invSurfaceAreaTarget;
        Emap3(particleVelOmega.data() + 6 * pidxTarget + 3) = invMomentOfInertiaTensorTarget * omega;
    }
}

// Y = alpha*A*x + beta*Y
template <class Container>
void HydroParticleOperator<Container>::measureb(const std::complex<double> *knownSurfaceDensityCoeffPtr,
                                                double *knownSurfaceDensityGridValsPtr, std::complex<double> *bCoeffPtr,
                                                double *bGridValsPtr, const double alpha = 1.0,
                                                const double beta = 0.0) const {
    const auto &particleContainer = *particleContainerPtr;

    // step 1: convert knownSurfaceDensityCoeff from spectral to grid
#pragma omp parallel for
    for (int pidx = 0; pidx < numPtcLocal; pidx++) {
        // particle properties
        const auto &ptc = particleContainer[pidx];
        const auto surface = ptc.getSurface();

        // reconstruct surfaceDensity from surfaceDensityCoeff
        const int pointidx = self.particlePtsIndex[pidx];
        const int coeffidx = self.particleCoeffIndex[pidx];

        const std::complex<double> *coeffVSHPtr = *(knownSurfaceDensityCoeffPtr + 3 * coeffidx);
        double *surfaceVectorPtr = *(knownSurfaceDensityGridValsPtr + 3 * pointidx);
        surface.reconstructSurfaceVectorFcn(coeffVSHPtr, surfaceVectorPtr);
    }

    // step 2: compute the operations on the grid values
    calcb(knownSurfaceDensityGridValsPtr, bGridValsPtr, alpha, beta);

    // step 3: convert from grid to spectral
#pragma omp parallel for
    for (int pidx = 0; pidx < numPtcLocal; pidx++) {
        // particle properties
        const auto &ptc = particleContainer[pidx];
        const auto surface = ptc.getSurface();

        // convert the grid values to spectral coeff
        const int pointidx = self.particlePtsIndex[pidx];
        const int coeffidx = self.particleCoeffIndex[pidx];

        const double *surfaceVectorPtr = *(bGridVals + 3 * pointidx);
        std::complex<double> *coeffVSHPtr = *(bCoeff + 3 * coeffidx);
        surface.decomposeSurfaceVectorFcn(surfaceVectorPtr, coeffVSHPtr);
    }
}

// Y := alpha A x + beta Y
template <class Container>
void HydroParticleOperator<Container>::measureAx(const std::complex<double> *unknownSurfaceDensityCoeffPtr,
                                                 double *unknownSurfaceDensityGridValsPtr,
                                                 std::complex<double> *AxCoeffPtr, double *AxGridValsPtr,
                                                 const double alpha = 1.0, const double beta = 0.0) const {
    const auto &particleContainer = *particleContainerPtr;

    // step 1: convert knownSurfaceDensityCoeff from spectral to grid
#pragma omp parallel for
    for (int pidx = 0; pidx < numPtcLocal; pidx++) {
        // particle properties
        const auto &ptc = particleContainer[pidx];
        const auto surface = ptc.getSurface();

        // reconstruct surfaceDensity from surfaceDensityCoeff
        const int coeffidx = self.particleCoeffIndex[pidx];
        const int pointidx = self.particlePtsIndex[pidx];

        const std::complex<double> *coeffVSHPtr = *(unknownSurfaceDensityCoeffPtr + 3 * coeffidx);
        double *surfaceVectorPtr = *(unknownSurfaceDensityGridValsPtr + 3 * pointidx);
        surface.reconstructSurfaceVectorFcn(coeffVSHPtr, surfaceVectorPtr);
    }

    // step 2: compute the operations on the grid values
    calcAx(unknownSurfaceDensityGridValsPtr, AxGridValsPtr, alpha, beta);

    // step 3: convert from grid to spectral
#pragma omp parallel for
    for (int pidx = 0; pidx < numPtcLocal; pidx++) {
        // particle properties
        const auto &ptc = particleContainer[pidx];
        const auto surface = ptc.getSurface();

        // convert the grid values to spectral coeff
        const int pointidx = self.particlePtsIndex[pidx];
        const int coeffidx = self.particleCoeffIndex[pidx];

        const double *surfaceVectorPtr = *(AxGridValsPtr + 3 * pointidx);
        std::complex<double> *coeffVSHPtr = *(AxCoeffPtr + 3 * coeffidx);
        surface.decomposeSurfaceVectorFcn(surfaceVectorPtr, coeffVSHPtr);
    }
}

// Y = alpha*A*x + beta*Y
template <class Container>
void HydroParticleOperator<Container>::measureBx(const double *totalForceTorquePtr, std::complex<double> *BxCoeffPtr,
                                                 double *BxGridValsPtr, double alpha = 1.0, double beta = 1.0) const {
    const auto &particleContainer = *particleContainerPtr;

    // step 1: compute Bx from the total force and torque
    calcBx(totalForceTorquePtr, BxGridValsPtr, alpha, beta);

    // step 2: convert from grid to spectral
#pragma omp parallel for
    for (int pidx = 0; pidx < numPtcLocal; pidx++) {
        // particle properties
        const auto &ptc = particleContainer[pidx];
        const auto surface = ptc.getSurface();

        // measure densityFromInducedTotalForceAndTorque using VSH
        const int pointidx = self.particlePtsIndex[pidx];
        const int coeffidx = self.particleCoeffIndex[pidx];

        const double *surfaceVectorPtr = *(BxGridValsPtr + 3 * pointidx);
        std::complex<double> *coeffVSHPtr = *(BxCoeffPtr + 3 * coeffidx);
        surface.decomposeSurfaceVectorFcn(surfaceVectorPtr, coeffVSHPtr);
    }
}

// Y = alpha*A*x + beta*Y
template <class Container>
void HydroParticleOperator<Container>::measureLx(const std::complex<double> *surfaceDensityCoeffPtr,
                                                 double *surfaceDensityGridValsPtr, std::complex<double> *LxCoeffPtr,
                                                 double *LxGridValsPtr, const double alpha = 1.0,
                                                 const double beta = 0.0) const {
    const auto &particleContainer = *particleContainerPtr;

    // step 1: convert surfaceDensityCoeffPtr from spectral to grid
#pragma omp parallel for
    for (int pidx = 0; pidx < numPtcLocal; pidx++) {
        // particle properties
        const auto &ptc = particleContainer[pidx];
        const auto surface = ptc.getSurface();

        // reconstruct surfaceDensity from surfaceDensityCoeff
        const int coeffidx = self.particleCoeffIndex[pidx];
        const int pointidx = self.particlePtsIndex[pidx];

        const std::complex<double> *coeffVSHPtr = *(surfaceDensityCoeffPtr + 3 * coeffidx);
        double *surfaceVectorPtr = *(surfaceDensityGridValsPtr + 3 * pointidx);
        surface.reconstructSurfaceVectorFcn(coeffVSHPtr, surfaceVectorPtr);
    }

    // step 2: compute the operations on the grid values
    calcLx(surfaceDensityGridValsPtr, LxCoeffPtr, alpha, beta);

    // step 3: convert from grid to spectral
#pragma omp parallel for
    for (int pidx = 0; pidx < numPtcLocal; pidx++) {
        // particle properties
        const auto &ptc = particleContainer[pidx];
        const auto surface = ptc.getSurface();

        // convert the grid values to spectral coeff
        const int pointidx = self.particlePtsIndex[pidx];
        const int coeffidx = self.particleCoeffIndex[pidx];

        const double *surfaceVectorPtr = *(LxGridValsPtr + 3 * pointidx);
        std::complex<double> *coeffVSHPtr = *(LxCoeffPtr + 3 * coeffidx);
        surface.decomposeSurfaceVectorFcn(surfaceVectorPtr, coeffVSHPtr);
    }
}

// Y = alpha*A*x + beta*Y
template <class Container>
void HydroParticleOperator<Container>::calcAx(const double *unknownSurfaceDensityGridValsPtr, double *AxGridValsPtr,
                                              const double alpha = 1.0, const double beta = 0.0) const {
    calcLx(unknownSurfaceDensityGridValsPtr, AxGridValsPtr, alpha, beta);
    calcJx(unknownSurfaceDensityGridValsPtr, AxGridValsPtr, alpha, 1.0);
}

// Y = alpha*A*x + beta*Y
template <class Container>
void HydroParticleOperator<Container>::calcb(const double *knownSurfaceDensityGridValsPtr, const double alpha = 1.0,
                                             const double beta = 0.0) const {
    return calcJx(knownSurfaceDensityGridValsPtr, -alpha, beta);
}

// Y = alpha*A*x + beta*Y
template <class Container>
void HydroParticleOperator<Container>::calcBx(const double *totalForceTorquePtr, double *BxGridValsPtr,
                                              const double alpha = 1.0, const double beta = 0.0) const {
    const auto &particleContainer = *particleContainerPtr;

#pragma omp parallel for
    for (int pidx = 0; pidx < numPtcLocal; pidx++) {
        // particle properties
        const auto &ptc = particleContainer[pidx];
        const auto surface = ptc.getSurface();
        const int numGridPts = surface.getNumGridPts();
        const Evec3 &centroid = ptc.pos;
        const Emat3 invMomentOfInertiaTensor = surface.getInvMomentOfInertiaTensorCurrentConfig();
        const double invSurfaceArea = 1.0 / surface.getSurfaceArea();

        // calculate the total force and torque induced by surfaceDensity
        const Evec3 totalForce = ECmap3(totalForceTorquePtr + 6 * pidx);
        const Evec3 totalTorque = ECmap3(totalForceTorquePtr + 6 * pidx + 3);

        // distribute the total force and torque to the entire surface
        const Evec3 tauInvT = invMomentOfInertiaTensor * totalTorque;
        const int pointidx = self.particlePtsIndex[pidx];
        for (int j = 0; j < numGridPts; j++) {
            const Evec3 gridPoint = surface.getGridPointCurrentConfig(ptc.pos, j);
            const int idx = 3 * (pointidx + j);
            BxGridValsPtr[idx + 0] = BxGridValsPtr[idx + 0] * beta + alpha * totalForce[0] * invSurfaceArea;
            BxGridValsPtr[idx + 1] = BxGridValsPtr[idx + 1] * beta + alpha * totalForce[1] * invSurfaceArea;
            BxGridValsPtr[idx + 2] = BxGridValsPtr[idx + 2] * beta + alpha * totalForce[2] * invSurfaceArea;
            BxGridValsPtr[idx] = BxGridValsPtr[idx] * beta + alpha * (tauInvT[1] * (gridPoint[2] - centroid[2]) -
                                                                      tauInvT[2] * (gridPoint[1] - centroid[1]));
            BxGridValsPtr[idx] = BxGridValsPtr[idx] * beta + alpha * (tauInvT[2] * (gridPoint[0] - centroid[0]) -
                                                                      tauInvT[0] * (gridPoint[2] - centroid[2]));
            BxGridValsPtr[idx] = BxGridValsPtr[idx] * beta + alpha * (tauInvT[0] * (gridPoint[1] - centroid[1]) -
                                                                      tauInvT[1] * (gridPoint[0] - centroid[0]));
        }
    }
}

// Y = alpha*A*x + beta*Y
template <class Container>
void HydroParticleOperator<Container>::calcLx(const double *surfaceDensityGridValsPtr, double *LxGridValsPtr,
                                              const double alpha = 1.0, const double beta = 0.0) const {
    const auto &particleContainer = *particleContainerPtr;

#pragma omp parallel for
    for (int pidx = 0; pidx < numPtcLocal; pidx++) {
        // particle properties
        const auto &ptc = particleContainer[pidx];
        const auto surface = ptc.getSurface();
        const int numGridPts = surface.getNumGridPts();
        const Evec3 &centroid = ptc.pos;
        const Emat3 invMomentOfInertiaTensor = surface.getInvMomentOfInertiaTensorCurrentConfig();
        const double invSurfaceArea = 1.0 / surface.getSurfaceArea();

        // calculate the total force and torque induced by surfaceDensityGridVals
        Evec3 inducedTotalForce(0.0, 0.0, 0.0);
        Evec3 inducedTotalTorque(0.0, 0.0, 0.0);
        const int ptsidx = self.particlePtsIndex[pidx];
        for (int j = 0; j < numGridPts; j++) {
            const Evec3 gridPoint = surface.getGridPointCurrentConfig(ptc.pos, j);

            inducedTotalForce[0] += surfaceDensityGridValsPtr[3 * (ptsidx + j) + 0] * surface.getGridWeight(j);
            inducedTotalForce[1] += surfaceDensityGridValsPtr[3 * (ptsidx + j) + 1] * surface.getGridWeight(j);
            inducedTotalForce[2] += surfaceDensityGridValsPtr[3 * (ptsidx + j) + 2] * surface.getGridWeight(j);
            inducedTotalTorque[0] += ((gridPoint[1] - centroid[1]) * surfaceDensityGridValsPtr[3 * (ptsidx + j) + 2] -
                                      (gridPoint[2] - centroid[2]) * surfaceDensityGridValsPtr[3 * (ptsidx + j) + 1]) *
                                     surface.getGridWeight(j);
            inducedTotalTorque[1] += ((gridPoint[2] - centroid[2]) * surfaceDensityGridValsPtr[3 * (ptsidx + j) + 0] -
                                      (gridPoint[0] - centroid[0]) * surfaceDensityGridValsPtr[3 * (ptsidx + j) + 2]) *
                                     surface.getGridWeight(j);
            inducedTotalTorque[2] += ((gridPoint[0] - centroid[0]) * surfaceDensityGridValsPtr[3 * (ptsidx + j) + 1] -
                                      (gridPoint[1] - centroid[1]) * surfaceDensityGridValsPtr[3 * (ptsidx + j) + 0]) *
                                     surface.getGridWeight(j);
        }

        // distribute the total force and torque to the entire surface
        const Evec3 tauInvT = invMomentOfInertiaTensor * inducedTotalTorque;
        for (int j = 0; j < numGridPts; j++) {
            const Evec3 gridPoint = surface.getGridPointCurrentConfig(ptc.pos, j);

            const int idx = 3 * (ptsidx + j);
            LxGridValsPtr[idx + 0] = beta * LxGridValsPtr[idx + 0] + alpha * inducedTotalForce[0] * invSurfaceArea;
            LxGridValsPtr[idx + 1] = beta * LxGridValsPtr[idx + 1] + alpha * inducedTotalForce[1] * invSurfaceArea;
            LxGridValsPtr[idx + 2] = beta * LxGridValsPtr[idx + 2] + alpha * inducedTotalForce[2] * invSurfaceArea;
            LxGridValsPtr[idx + 0] =
                beta * LxGridValsPtr[idx + 0] +
                alpha * (tauInvT[1] * (gridPoint[2] - centroid[2]) - tauInvT[2] * (gridPoint[1] - centroid[1]));
            LxGridValsPtr[idx + 1] =
                beta * LxGridValsPtr[idx + 1] +
                alpha * (tauInvT[2] * (gridPoint[0] - centroid[0]) - tauInvT[0] * (gridPoint[2] - centroid[2]));
            LxGridValsPtr[idx + 2] =
                beta * LxGridValsPtr[idx + 2] +
                alpha * (tauInvT[0] * (gridPoint[1] - centroid[1]) - tauInvT[1] * (gridPoint[0] - centroid[0]));
        }
    }
}

// Y = alpha*A*x + beta*Y
template <class Container>
void HydroParticleOperator<Container>::calcJx(const double *surfaceDensityGridValsPtr, double *JxGridValsPtr,
                                              const double alpha = 1.0, const double beta = 0.0) const {
    const auto &particleContainer = *particleContainerPtr;

#pragma omp parallel for
    for (int pidxTarget = 0; pidxTarget < numPtcLocal; pidxTarget++) {
        // particle properties
        const auto &ptcTarget = particleContainer[pidx];
        const auto surfaceTarget = ptcTarget.getSurface();
        const int numGridPtsTarget = surfaceTarget.getNumGridPts();
        const int ptsidxTarget = self.particlePtsIndex[pidxTarget];

        double *JTarget = *(JxGridValsPtr + 3 * ptsidxTarget);
        for (int i = 0; i < numGridPtsTarget; i++) {
            const Evec3 gridPointTargeti = surfaceTarget.getGridPointCurrentConfig(ptcTarget.pos, i);
            const Evec3 gridNormTargeti = surfaceTarget.getGridNormCurrentConfig(i);
            const Evec3 gridDensityTargeti = ECmap3(surfaceDensityGridValsPtr + 3 * (ptsidxTarget + i));

            for (int pidxSource = 0; pidxSource < numPtcLocal; pidxSource++) {
                const bool selfInteraction = pidxTarget == pidxSource ? true : false;

                // particle properties
                const auto &ptcSource = particleContainer[pidx];
                const auto surfaceSource = ptcSource.getSurface();
                const int numGridPtsSource = surfaceSource.getNumGridPts();
                const int ptsidxSource = self.particlePtsIndex[pidxSource];

                for (int j = 0; j < numGridPtsSource; j++) {
                    // singular subtraction allows us to skip the singular self-interaction component
                    if (selfInteraction) {
                        if (i == j) {
                            continue;
                        }
                    }

                    const Evec3 gridPointSourcej = surfaceSource.getGridPointCurrentConfig(ptcSource.pos, j);
                    const Evec3 gridNormSourcej = surfaceSource.getGridNormCurrentConfig(j);
                    const Evec3 gridDensitySourcej = ECmap3(surfaceDensityGridValsPtr + 3 * (ptsidxSource + j));

                    const Evec3 rVec = gridPointTargeti - gridPointSourcej;
                    const Evec3 tractionVec =
                        self.stresslet(rVec, gridDensityTargeti, gridDensitySourcej, gridNormTargeti, gridNormSourcej);
                    Emap3(JTarget + 3 * i) =
                        beta * Emap3(JTarget + 3 * i) + alpha * tractionVec * surfaceSource.getGridWeight(j);
                }
            }
        }
    }
}

template <class Container>
void HydroParticleOperator<Container>::getUserInputSurfaceDensityCoeffWrapper(TV &userInputSurfaceDensityCoeffVec,
                                                                              const double alpha = 1.0,
                                                                              const double beta = 0.0) const {
    const int nRows = userInputSurfaceDensityCoeff.getLocalLength();

    // get the local views
    auto userInputSurfaceDensityCoeffVecPtr = userInputSurfaceDensityCoeffVec.getLocalView<Kokkos::HostSpace>();

    // copy the kokkos local view into an std::vector for ease of manipulation
    std::vector<std::complex<double>> userInputSurfaceDensityCoeff(nRows);
#pragma omp parallel for
    for (int i = 0; i < nRows; i++) {
        userInputSurfaceDensityCoeff[i] = userInputSurfaceDensityCoeffVecPtr(i);
    }

    // preform the computation
    getUserInputSurfaceDensityCoeff(userInputSurfaceDensityCoeff.data(), alpha, beta);

    // copy the results into the kokkos local view
#pragma omp parallel for
    for (int i = 0; i < nRows; i++) {
        userInputSurfaceDensityCoeffVecPtr(i) += userInputSurfaceDensityCoeffVecPtr[i];
    }
}

template <class Container>
void HydroParticleOperator<Container>::getSurfaceDensityCoeffFromParticleContainerWrapper(
    TV &surfaceDensityCoeffVec, const double alpha = 1.0, const double beta = 0.0) const {
    const auto &particleContainer = *particleContainerPtr;

#pragma omp parallel for
    for (int pidx = 0; pidx < numPtcLocal; pidx++) {
        // particle properties
        const auto &ptc = particleContainer[pidx];
        const auto surface = ptc.getSurface();
        const int numGridPts = surface.getNumGridPts();

        // reconstruct surfaceDensity from surfaceDensityCoeff
        const int pointidx = self.particlePtsIndex[pidx];
        const int coeffidx = self.particleCoeffIndex[pidx];

        for (int i = 0; i < numGridPts; i++) {
            surfaceDensityCoeffVec(3 * (coeffidx + i) + 0) =
                beta * surfaceDensityCoeffVec(3 * (coeffidx + i) + 0) + alpha * layer.surfaceDensityCoeff[3 * i + 0];
            surfaceDensityCoeffVec(3 * (coeffidx + i) + 1) =
                beta * surfaceDensityCoeffVec(3 * (coeffidx + i) + 1) + alpha * layer.surfaceDensityCoeff[3 * i + 1];
            surfaceDensityCoeffVec(3 * (coeffidx + i) + 2) =
                beta * surfaceDensityCoeffVec(3 * (coeffidx + i) + 2) + alpha * layer.surfaceDensityCoeff[3 * i + 2];
        }
    }
}

template <class Container>
void HydroParticleOperator<Container>::measurebWrapper(const TV &knownSurfaceDensityCoeffVec, TV &bCoeffVec,
                                                       const double alpha = 1.0, const double beta = 0.0) const {
    const int nRowDomain = knownSurfaceDensityCoeffVec.getLocalLength();
    const int nRowRange = bCoeffVec.getLocalLength();
    assert(nRowDomain == surfaceDensityCoeff.size());
    assert(nRowRange == surfaceDensityCoeff.size());

    // get the local views
    auto knownSurfaceDensityCoeffVecPtr = knownSurfaceDensityCoeffVec.getLocalView<Kokkos::HostSpace>();
    auto bCoeffVecPtr = bCoeffVec.getLocalView<Kokkos::HostSpace>();

    // copy the kokkos local view into an std::vector for ease of manipulation
#pragma omp parallel for
    for (int i = 0; i < nRowDomain; i++) {
        knownSurfaceDensityCoeff[i] += knownSurfaceDensityCoeffVecPtr(i);
    }

    // preform the computation
    measureb(unknownSurfaceDensityCoeff.data(), knownSurfaceDensityGridVals.data(), bCoeff.data(), bGridVals.data(),
             alpha, beta);

    // copy the results into the kokkos local view
#pragma omp parallel for
    for (int i = 0; i < nRowRange; i++) {
        bCoeffVecPtr(i) += AxCoeff[i];
    }
}

template <class Container>
void HydroParticleOperator<Container>::measureAxWrapper(const TV &unknownSurfaceDensityCoeffVec, TV &AxCoeffVec) const {
    const int nRowDomain = unknownSurfaceDensityCoeffVec.getLocalLength();
    const int nRowRange = AxCoeffVec.getLocalLength();
    assert(nRowDomain == unknownSurfaceDensityCoeff.size());
    assert(nRowRange == AxCoeff.size());

    // get the local views
    auto unknownSurfaceDensityCoeffVecPtr = unknownSurfaceDensityCoeffVec.getLocalView<Kokkos::HostSpace>();
    auto AxCoeffVecPtr = AxCoeffVec.getLocalView<Kokkos::HostSpace>();

    // copy the kokkos local view into an std::vector for ease of manipulation
#pragma omp parallel for
    for (int i = 0; i < nRowDomain; i++) {
        unknownSurfaceDensityCoeff[i] += unknownSurfaceDensityCoeffVecPtr(i);
    }

    // preform the computation
    measureAx(unknownSurfaceDensityCoeff.data(), unknownSurfaceDensityGridVals.data(), AxCoeff.data(),
              AxGridVals.data());

    // copy the results into the kokkos local view
#pragma omp parallel for
    for (int i = 0; i < nRowRange; i++) {
        AxCoeffVecPtr(i) += AxCoeff[i];
    }
}

template <class Container>
void HydroParticleOperator<Container>::measureBxWrapper(const TV &totalForceTorqueVec, TV &BxCoeffVec) const {
    const int nRowDomain = totalForceTorqueVec.getLocalLength();
    const int nRowRange = BxCoeffVec.getLocalLength();
    assert(nRowDomain == totalExtForceTorque.size());
    assert(nRowRange == BxCoeff.size());

    // get the local views
    auto totalForceTorqueVecPtr = totalForceTorqueVec.getLocalView<Kokkos::HostSpace>();
    auto BxCoeffVecPtr = BxCoeffVec.getLocalView<Kokkos::HostSpace>();

    // copy the kokkos local view into an std::vector for ease of manipulation
#pragma omp parallel for
    for (int i = 0; i < nRowDomain; i++) {
        totalForceTorqueVec[i] += totalForceTorqueVecPtr(i);
    }
    
    // preform the computation
    measureBx(totalForceTorqueVecPtr.data(), BxCoeff.data(), BxGridVals.data(),
              BxGridVals.data());

    // copy the results into the kokkos local view
#pragma omp parallel for
    for (int i = 0; i < nRowRange; i++) {
        BxCoeffVecPtr(i) += BxCoeff[i];
    }
}


template <class Container>
void HydroParticleOperator<Container>::apply(const TMV &X, TMV &Y, Teuchos::ETransp mode, scalar_type alpha,
                                             scalar_type beta) const {
    // compute Y=alpha*Ax+beta*Y;
    assert(X.getMap()->isSameAs(*(Y.getMap())));
    assert(X.getMap()->isSameAs(*pointValuesMapRcp));
    assert(X.getNumVectors() == Y.getNumVectors());

    Teuchos::RCP<TV> YColOld = Teuchos::rcp(new TV(Y.getMap(), true));

    const int nCol = X.getNumVectors();
    for (int c = 0; c < nCol; c++) {
        const auto &XCol = X.getVector(c);
        auto YCol = Y.getVectorNonConst(c);
        YColOld->update(beta, *YCol, Teuchos::ScalarTraits<scalar_type>::zero()); // Yold = beta*Ycol
        measureAxWrapper(*XCol, *YCol);                                           // Ycol = AXcol
        YCol->update(Teuchos::ScalarTraits<scalar_type>::one(), *YColOld, alpha); // Ycol = alpha*AXcol+beta*Ycol
    }
}
