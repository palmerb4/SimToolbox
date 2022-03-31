#include "ParticleSurface.hpp"

constexpr double pi = 3.1415926535897932384626433;

/*****************************************************
 *  Genus zero particle surface
 *****************************************************/

template <int maxOrder>
void ParticleSurface<maxOrder>::clear() {       
    EmatmapNS(surfaceDensityGridVals).setZero();
    EmatmapNS(surfaceVelocityGridVals).setZero();
    EmatmapNS(surfaceDensityCoeff).setZero();
    EmatmapNS(surfaceVelocityCoeff).setZero();
}

template <int maxOrder>
void ParticleSurface<maxOrder>::storeSharedSurface(const SharedParticleSurface<maxOrder> *SharedPSPtr) {
    this->SharedPS = std::ref(&SharedPSPtr);
}

template <int maxOrder>
const int ParticleSurface<maxOrder>::getNumGridPts() const {
    return sharedPS.getNumGridPts();
}

template <int maxOrder>
const int ParticleSurface<maxOrder>::getNumSpectralCoeff() const {
    return sharedPS.getNumSpectralCoeff();
}

template <int maxOrder>
const Evec3 ParticleSurface<maxOrder>::getGridPointCurrentConfig(const Evec3 coordBase, const int idx) const {
    return sharedPS.getGridPointCurrentConfig(coordBase, orientation, idx);
}

template <int maxOrder>
const Evec3 ParticleSurface<maxOrder>::getGridNormCurrentConfig(const int idx) const {
    return sharedPS.getGridNormCurrentConfig(orientation, idx);
}

template <int maxOrder>
const Emat3 ParticleSurface<maxOrder>::getInvMomentOfInertiaTensorCurrentConfig() const {
    return sharedPS.getInvMomentOfInertiaTensorCurrentConfig(orientation)
}

template <int maxOrder>
const double ParticleSurface<maxOrder>::getGridWeight(const int idx) const {
    return sharedPS.getGridWeight(idx);
}

template <int maxOrder>
const double ParticleSurface<maxOrder>::getSurfaceArea() const {
    return sharedPS.getSurfaceArea();
}

template <int maxOrder>
void ParticleSurface<maxOrder>::decomposeSurfaceVectorFcn(const double *vecSurface, std::complex<double> *vshCoeff) const {
    EmatmapNS3(vshCoeff) = sharedPS.decomposeSurfaceVectorFcn(ECmatmapNP3(vecSurface), orientation);
}

template <int maxOrder>
void ParticleSurface<maxOrder>::reconstructSurfaceVectorFcn(const std::complex<double> *vshCoeff, double *vecSurface) const {
    EmatmapNP3(vecSurface) = sharedPS.reconstructSurfaceVectorFcn(ECmatmapNS3(vshCoeff), orientation);
}

template <int maxOrder>
void ParticleSurface<maxOrder>::decomposeSurfaceScalarFcn(const double *scalarSurface, std::complex<double> *shCoeff) const {
    EmatmapNS(shCoeff) = sharedPS.decomposeSurfaceScalarFcn(ECmatmapNP(scalarSurface));
}

template <int maxOrder>
void ParticleSurface<maxOrder>::reconstructSurfaceScalarFcn(const std::complex<double> *shCoeff, double *scalarSurface) const {
    EmatmapNP(scalarSurface) = sharedPS.reconstructSurfaceScalarFcn(ECmatmapNS(shCoeff));
}

template <int maxOrder>
int ParticleSurface<maxOrder>::writeVTU(std::ofstream &file, const Evec3 &coordBase) const {
    // indexBase is the index of the first grid point

    // this must be called in single thread
    // change to using cpp to avoid copy from string to c_str()

    assert(file.is_open());
    assert(file.good());

    std::vector<double> gridPoints;
    std::vector<double> gridWeights;
    getGridWithPole(gridPoints, gridWeights, coordBase);
    const int nPts = gridWeights.size();
    assert(gridPoints.size() == nPts * 3);

// for debug
#ifdef DEBUGVTU
    printf("%lu,%lu,%lu\n", gridPoints.size(), gridWeights.size(), gridValue.size());
    for (const auto &v : gridPoints) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
    for (const auto &v : gridWeights) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
#endif

    std::string contentB64; // data converted to base64 format
    contentB64.reserve(10 * order * order * (kind == KIND::LAP ? 1 : 3));
    contentB64.clear();

    std::vector<int32_t> connect;
    std::vector<int32_t> offset;
    std::vector<uint8_t> cellTypes;
    getGridWithPoleCellConnect(connect, offset, cellTypes);

    std::vector<double> gridValueWithPole(getGridDOF() + 2 * getDimension());
    std::vector<double> spectralValues(getSpectralDOF());

    calcSpectralCoeff(spectralValues.data());
    double poleValues[6] = {0, 0, 0, 0, 0, 0};
    calcPoleValue(spectralValues.data(), poleValues);
    // put the pole values in the beginning and end of the array
    if (kind == KIND::LAP) {
        gridValueWithPole[0] = poleValues[0];
        std::copy(gridValue.cbegin(), gridValue.cend(), gridValueWithPole.begin() + 1);
        gridValueWithPole.back() = poleValues[1];
    } else {
        gridValueWithPole[0] = poleValues[0];
        gridValueWithPole[1] = poleValues[1];
        gridValueWithPole[2] = poleValues[2];
        std::copy(gridValue.cbegin(), gridValue.cend(), gridValueWithPole.begin() + 3);
        gridValueWithPole[3 + gridValue.size()] = poleValues[3];
        gridValueWithPole[4 + gridValue.size()] = poleValues[4];
        gridValueWithPole[5 + gridValue.size()] = poleValues[5];
        // printf("north pole: %lf,%lf,%lf\n", poleValues[0], poleValues[1], poleValues[2]);
        // printf("south pole: %lf,%lf,%lf\n", poleValues[3], poleValues[4], poleValues[5]);
    }

// for debug
#ifdef DEBUGVTU
    for (const auto &v : connect) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
    for (const auto &v : offset) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
    for (const auto &v : cellTypes) {
        int temp = v;
        std::cout << temp << " ";
    }
    std::cout << std::endl;
#endif

    // write a vtk unstructured grid section
    // assume file in 'append' mode

    const int p = order;
    file << "<Piece NumberOfPoints=\"" << 2 * p * p + 4 * p + 4 << "\" NumberOfCells=\"" << (2 * p + 2) * (p + 2)
         << "\">\n";

    // point data
    file << "<PointData Scalars=\"scalars\">\n";
    IOHelper::writeDataArrayBase64(gridValueWithPole, this->name, ((kind == KIND::LAP) ? 1 : 3), file);
    IOHelper::writeDataArrayBase64(gridWeights, "weights", 1, file);
    file << "</PointData>\n";

    // cell data (empty)

    // point location
    file << "<Points>\n";
    IOHelper::writeDataArrayBase64(gridPoints, "points", 3, file);
    file << "</Points>\n";

    // cell definition
    file << "<Cells>\n";
    IOHelper::writeDataArrayBase64(connect, "connectivity", 1, file);
    IOHelper::writeDataArrayBase64(offset, "offsets", 1, file);
    IOHelper::writeDataArrayBase64(cellTypes, "types", 1, file);
    file << "</Cells>\n";

    // end
    file << "</Piece>" << std::endl; // flush

    return nPts;
}


template <int maxOrder>
void ParticleSurface<maxOrder>::calcGridCellConnect(std::vector<int32_t> &gridCellConnect, std::vector<int32_t> &offset,
                                    std::vector<uint8_t> &cellTypes) const {
    // Note that offset gives the END position of each cell

    const int p = order;
    // for p=0 two points on equator. Define 4 3-node cells

    // cells with north pole, 3 point for each cell
    int index = 0; // the index to the node point, starting from 1 point after the north pole
    for (int k = 0; k < 2 * p + 1; k++) {
        // 3 points, 0,k,k+1
        index++;
        gridCellConnect.push_back(0);
        gridCellConnect.push_back(index);
        gridCellConnect.push_back(index + 1);
        cellTypes.push_back(uint8_t(5)); // 5= VTK_TRIANGLE
        offset.push_back(gridCellConnect.size());
    }
    index++;
    gridCellConnect.push_back(0);
    gridCellConnect.push_back(2 * p + 2);
    gridCellConnect.push_back(1);
    cellTypes.push_back(uint8_t(5)); // 5= VTK_TRIANGLE
    offset.push_back(gridCellConnect.size());

    // 4 cells for each cell in the center
    for (int j = 1; j < p + 1; j++) {
        for (int k = 0; k < 2 * p + 1; k++) {
            // 4 points, index, index+1, index-(2p+2), index+1 - (2p+2)
            index++;
            gridCellConnect.push_back(index);
            gridCellConnect.push_back(index + 1);
            gridCellConnect.push_back(index + 1 - (2 * p + 2));
            gridCellConnect.push_back(index - (2 * p + 2));
            cellTypes.push_back(uint8_t(9)); // 9 = VTK_QUAD
            offset.push_back(gridCellConnect.size());
        }
        // last one, connect to the first one in this circle
        index++;
        gridCellConnect.push_back(index);
        gridCellConnect.push_back(index - (2 * p + 1));
        gridCellConnect.push_back(index - (2 * p + 1) - (2 * p + 2));
        gridCellConnect.push_back(index - (2 * p + 2));
        cellTypes.push_back(uint8_t(9)); // 9 = VTK_QUAD
        offset.push_back(gridCellConnect.size());
    }

    // cells with south pole, 3 points for each cell
    gridCellConnect.push_back(2 * p * p + 4 * p + 3);               // index for south pole
    gridCellConnect.push_back(2 * p * p + 4 * p + 2);               // 1 before south pole
    gridCellConnect.push_back(2 * p * p + 4 * p + 2 - (2 * p + 1)); // 1 around the circle
    cellTypes.push_back(uint8_t(5));                                     // 5= VTK_TRIANGLE
    offset.push_back(gridCellConnect.size());
    for (int k = 0; k < 2 * p + 1; k++) {
        // 3 points, k,k+1, southpole
        index--;
        gridCellConnect.push_back(2 * p * p + 4 * p + 3); // index for south pole
        gridCellConnect.push_back(index + 1);
        gridCellConnect.push_back(index);
        cellTypes.push_back(uint8_t(5)); // 5= VTK_TRIANGLE
        offset.push_back(gridCellConnect.size());
    }
}

template <int maxOrder>
EmatNS3 ParticleSurface<maxOrder>::getUserInputSurfaceDensity(const string prescribedSurfaceDensityFile) const {

}

