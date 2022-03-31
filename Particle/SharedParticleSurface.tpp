#include "SharedParticleSurface.hpp"

#include "Util/Gauss_Legendre_Nodes_and_Weights.hpp"
#include "Util/rapidcsv.h"

constexpr double pi = 3.1415926535897932384626433;

/*********************************************
 *  Precomputed genus zero particle surface  *
 *********************************************/

// Note: The use of const within the following code is done strategically
//       to ensure  that return value optimization works as intended,
//       as to avoid unnecessary copies of large Eigen objects.

// TODO
//     1. (Done) Replace Rotation with something from Eigen
//     2. (Done) Switch double to std::complex<double>
//     3. (Done) Optimize for return value optimization
//     4. (Done) Is decomposeSurfaceScalarFcn correct? Can we ignore the jacobian and preform the integral on the
//     sphere, rather than the surface? Yes.
//     5. (Done) Delete all refs to basis B. We only precompute for basis A.
//     6. Make sure the type casting between complex and real is done correctly
//     7. (Done) Overload the inner products to work with real/complex and scalar/vector
//     8. (Done) Figure out how to construct the pushforward/pullback and how to map to and from their domain/range
//     space
//     9. (Done) Add calcThetaPhiWeights to this class.
//     10. (Done) add a means for storing and reading from the prescibed surface density csv file

template <int spectralDegree>
void SharedParticleSurface<spectralDegree>::SharedParticleSurface(const std::string &name,
                                                                  const std::shared_ptr<const ParticleShapes> &particleShapePtr, const Evec3 north,
                                                                  const std::string prescribedSurfaceDensityFile) {
    this->name = name;
    this->particleShapesPtr =
        particleShapesPtr; // TODO: Does this create a copy? 
    this->north = north;
    this->prescribedSurfaceDensityFile = prescribedSurfaceDensityFile;
    precompute();
}

template <int spectralDegree>
void SharedParticleSurface<spectralDegree>::precompute() {
    // the names speak for themselves
    precomputeGuassLegandreQuadrature();
    precomputeSurfaceProperties();
    precomputeSpectralHarmonics();
    if (prescribedSurfaceDensityFile.isempty()) {
        prescribedSurfaceDensityCoeff.fill(0.0);
    } else {
        storePrescribedSurfaceDensityCoeff();
    }
}

template <int spectralDegree>
void SharedParticleSurface<spectralDegree>::precomputeGuassLegandreQuadrature() {
    // Compute the locations and weights of the Gauss-Legendre quadrature for the sphere
    // rotated such that true north [0.0, 0.0, 1.0] is rotated to this->north
    //
    // The quadrature has
    // (p+1)(2p+2) = 2p^2+4p+2 total point without pole
    // (p+1)(2p+2) + north/south pole = 2p^2+4p+4 total point with pole
    //
    // Precomputes:
    //     np.array len N: thetasA, azimuthal angles with z-axis pole at each quadrature point
    //     np.array len N: phisA, polar angles with z-axis pole at each quadrature point
    //     np.array len N: thetasB, azimuthal angles with x-axis pole at each quadrature point
    //     np.array len N: phisB, polar angles with x-axis pole at each quadrature point
    //     np.array len N: sphereWeights, quadrature weights at each quadrature point on the sphere
    //

    Emat3 rotationMatAboutZAxis(const double psiA) {
        const Emat3 rotMat = {
            {std::cos(psiA), -std::sin(psiA), 0.}, {std::sin(psiA), std::cos(psiA), 0.}, {0., 0., 1.}};
        return rotMat;
    }

    Emat3 rotationMatAboutYAxis(const double psiA) {
        const Emat3 rotMat = {
            {std::cos(psiA), 0., std::sin(psiA)}, {0., 1., 0.}, {-std::sin(psiA), 0., std::cos(psiA)}};
        return rotMat;
    }

    Emat3 rotationMat(const double thetaA, const double phiA) {
        const Emat3 rotMat =
            rotationMatAboutZAxis(phiA) * rotationMatAboutYAxis(-thetaA) * rotationMatAboutZAxis(-phiA);
        return rotMat;
    }

    // step 1: compute the quadrature with typical north=[0.0, 0.0, 1.0]

    // p+1 points, excluding the two poles
    std::vector<double> nodesGL; // cos thetaj = tj
    std::vector<double> weightsGL;
    Gauss_Legendre_Nodes_and_Weights(spectralDegree + 1, nodesGL, weightsGL); // p+1 points, excluding the two poles

    // calculate sphere coordinate without rotation
    // between north and south pole
    // from north pole (1) to south pole (-1), picking the points from nodesGL in reversed order
    for (int j = 0; j < spectralDegree + 1; j++) {
        for (int k = 0; k < 2 * spectralDegree + 2; k++) {
            // set sphere values
            const int idx = (j * (2 * spectralDegree + 2)) + k + 1;

            // area element = sin thetaj
            const double weightfactor = 2 * pi / (2 * spectralDegree + 2);
            sphereWeights[idx] = weightfactor * weightsGL[spectralDegree - j];
            thetasA[idx] = std::acos(nodesGL[spectralDegree - j]);
            phisA[idx] = 2 * pi * k / (2 * spectralDegree + 2);
            thetasB[idx] = std::acos(std::sin(thetasA[idx]) * std::cos(phisA[idx]));
            phisB[idx] = std::atan2(std::sin(thetasA[idx]) * std::sin(phisA[idx]), std::cos(thetasA[idx]));
        }
    }

    // north pole
    sphereWeights[0] = 0.0;
    thetasA[0] = 0.0;
    phisA[0] = 0.0;
    thetasB[0] = pi / 2.0;
    phisB[0] = 0.0;

    // south pole
    sphereWeights[numGridPts - 1] = 0.0;
    thetasA[numGridPts - 1] = pi;
    phisA[numGridPts - 1] = 2.0 * pi;
    thetasB[numGridPts - 1] = np / 2.0;
    phisB[numGridPts - 1] = -pi;

    // step 2: rotate the quadrature to have north=this->north
    double thetaANorth;
    double phiANorth;
    particleShape.calcThetaPhiAFromPoint(north, thetaANorth, phiANorth);
    const Emat3 rotMat = rotationMat(thetaANorth, phiANorth);

    for (int i = 0; i < numGridPts + 1; i++) {
        const Evec3 gridPointUnRotated = particleShape.calcSurfacePointA(thetasA[i], phisA[i]);
        gridPointsRefConfig.row(i) = rotMat * gridPointUnRotated;
        particleShape.calcThetaPhiAFromPoint(ridPointsRefConfig.row(i), thetasA[i], phisA[i]);
        particleShape.calcThetaPhiBFromPoint(ridPointsRefConfig.row(i), thetasB[i], phisB[i]);
    }
}

template <int spectralDegree>
void SharedParticleSurface<spectralDegree>::precomputeSpectralHarmonics() {
    // These properties are all in the reference config.
    // surfaceNorm, surfacePoints, surface area, volume, centroid, moment of inertia tensor
    // pullback matrix, pushforward matrix

    // precompute the properties at each grid point
    for (int i = 0; i < numGridPts; i++) {
        particleShape.calcSurfaceNormalHatAndWeight(thetasA[i], phisA[i], thetasB[i], phisB[i], sphereWeights[idx],
                                                    gridNormsRefConfig.row(i), gridWeights[i]);
        particleShape.calcSurfacePointA(thetasA[i], phisA[i], gridPointsRefConfig.row(i));
        pullbackMatrix.block<3, 3>(3 * i, 3 * i) =
            self.particleShape.calcPullbackMatrix(thetasA[i], phisA[i], thetasB[i], phisB[i]);
        pushforwardMatrix.block<3, 3>(3 * i, 3 * i) =
            self.particleShape.calcPushforwardMatrix(thetasA[i], phisA[i], thetasB[i], phisB[i]);
    }

    // precompute SurfaceArea
    surfaceArea = gridWeights.sum();

    // precompute Centroid
    centroidRefConfig.fill(0.0);
    for (int i = 0; i < numGridPts; i++) {
        centroidRefConfig += gridPointsRefConfig.row(i) * gridWeights[i];
    }
    centroidRefConfig /= surfaceArea;

    // precompute MomentOfInertiaTensor
    invMomentOfInertiaTensorRefConfig.fill(0.0);
    for (int i = 0; i < numGridPts; i++) {
        const vec = gridPointsRefConfig.row(i) - centroidRefConfig;
        const double x = vec[0];
        const double y = vec[1];
        const double z = vec[2];
        invMomentOfInertiaTensorRefConfig += np.array([[y * y + z * y, -x * y, -x * z], [-x * y, z * z + x * x, -y * z],
                                                       [-x * z, -y * z, x * x + y * y]]) *
                                             gridWeights[i];
    }
}

template <int spectralDegree>
void SharedParticleSurface<spectralDegree>::precomputeSpectralProperties() {
    // precomputed vsh and sh at each grid point in the ref config
    for (int n = 0; n = < spectralDegree; n++) {
        for (int m = -n; m >= -n && m <= n; m++) {
            idxnm = n * n + m + n;
            EfieldNPcd &scalarSHnm = scalarSH[idxnm];
            EfieldNPcd &radialVSHnm = radialVSH[idxnm];
            EfieldNPcd &divfreeVSHnm = divfreeVSH[idxnm];
            EfieldNPcd &curlfreeVSHnm = curlfreeVSH[idxnm];

            for (int i = 0; i < numGridPts; i++) {
                self.scalarSHnm[idx] = spectral.getYnm(n, m, thetasA[i], phisA[i]);
                self.radialVSHnm.row(i) = spectral.getRadialVSHnmA(n, m, thetasA[i], phisA[i]);
                self.divfreeVSHnm.row(i) = spectral.getDivfreeVSHnmA(n, m, thetasA[i], phisA[i]);
                self.curlfreeVSHnm.row(i) = spectral.getCurlfreeVSHnmA(n, m, thetasA[i], phisA[i]);
            }
        }
    }
}

template <int spectralDegree>
int SharedParticleSurface<spectralDegree>::getNumGridPts() const {
    return numGridPts;
}

template <int spectralDegree>
int SharedParticleSurface<spectralDegree>::getNumSpectralCoeff() const {
    return numSpectralCoeff;
}

template <int spectralDegree>
Evec3 SharedParticleSurface<spectralDegree>::getGridPointCurrentConfig(const Evec3 &coordBase,
                                                                       const Equatn &orientation, const int idx) const {
    // shift the grid to be centered around the centroid
    const Evec3 &pointVec = gridPointsRefConfig.row(idx);
    const Evec3 shiftedPointVec = pointVec - centroidRefConfig;

    // rotation with quaternion
    Evec3 gridPointCurrentConfig = orientation.normalized().toRotationMatrix() * shiftedPointVec;

    // shift coordinate base
    Evec3 gridPointCurrentConfig += coordBase;
    return gridPointCurrentConfig;
}

template <int spectralDegree>
Evec3 SharedParticleSurface<spectralDegree>::getGridNormCurrentConfig(const Equatn &orientation, const int idx) const {
    // rotation with quaternion and then shift for each point coordinate
    const Evec3 &gridNorm = gridNormsRefConfig.row(idx);
    Evec3 gridNormCurrentConfig = orientation.normalized().toRotationMatrix() * gridNorm;
    return gridNormCurrentConfig;
}

template <int spectralDegree>
Emat3 SharedParticleSurface<spectralDegree>::getInvMomentOfInertiaTensorCurrentConfig(const Equatn &orientation) const {
    // inverse rotation matrix from quaternion
    // the conjugate of a quaternion represents the opposite rotation
    Emat3 R = orientation.normalized().toRotationMatrix();
    Emat3 Rinv = orientation.normalized().conjugate().toRotationMatrix();
    // apply transformation
    return Rinv * self.invMomentOfInertiaTensorRefConfig * R;
}

template <int spectralDegree>
double SharedParticleSurface<spectralDegree>::getGridWeight(const int idx) const {
    return gridWeights[idx];
}

template <int spectralDegree>
double SharedParticleSurface<spectralDegree>::getSurfaceArea() const {
    return surfaceArea;
}

template <int spectralDegree>
EfieldNP3 SharedParticleSurface<spectralDegree>::rotateSurfaceVecCurrentConfigToRefConfig(
    const EfieldNP3 &vecSurfaceCurrentConfig, const Equatn &orientation) const {
    EfieldNP3 vecSurfaceRef;

    Emat3 Rinv = orientation.normalized().conjugate().toRotationMatrix();
    for (int i = 0; i < numGridPts; i++) {
        vecSurfaceRef.row(i) = Rinv * vecSurfaceCurrentConfig.row(i);
    }
    return vecSurfaceRef;
}

template <int spectralDegree>
EfieldNP3 SharedParticleSurface<spectralDegree>::rotateSurfaceVecRefToCurrentConfig(const EfieldNP3 &vecSurfaceRef,
                                                                                    const Equatn &orientation) const {
    EfieldNP3 vecSurfaceCurrentConfig;

    Emat3 R = orientation.normalized().toRotationMatrix();
    for (int i = 0; i < numGridPts; i++) {
        vecSurfaceCurrentConfig.row(i) = R * vecSurfaceRef.row(i);
    }
    return vecSurfaceCurrentConfig;
}

template <int spectralDegree>
EfieldNP3 SharedParticleSurface<spectralDegree>::pullbackSurfaceVectors(const EfieldNP3 &vecSurface,
                                                                        const Equatn &orientation) const {
    // convert current config to the ref config
    vecSurfaceRef = rotateSurfaceVecCurrentConfigToRefConfig(vecSurface, orientation);

    // perform the pullback (operation is performed in place on the flattened vector)
    auto vecSurfaceRefFlat = vecSurfaceRef.reshaped<Eigen::RowMajor>();
    vecSurfaceRefFlat.applyOnTheLeft(vecSurfaceRefFlat);

    return vecSurfaceRef;
}

template <int spectralDegree>
EfieldNP3 SharedParticleSurface<spectralDegree>::pushforwardSphereVectors(const EfieldNP3 &vecSphere,
                                                                          const Equatn &orientation) const {
    // create a copy of vecSphere to allow for in place operations
    EfieldNP3 vecSphereCopy = vecSphere;

    // perform the pushforward (operation is performed in place on the flattened vector)
    auto vecSphereFlat = vecSphereCopy.reshaped<Eigen::RowMajor>();
    vecSphereFlat.applyOnTheLeft(pushforwardMatrix);

    // convert ref config to current config
    return rotateSurfaceVecRefToCurrentConfig(vecSphereCopy, orientation);
}

template <int spectralDegree>
EmatNS3 SharedParticleSurface<spectralDegree>::decomposeSurfaceVectorFcn(const EmatNP3 &vecSurface,
                                                                         const Equatn &orientation) const {
    // defaults to Basis A
    return decomposeSurfaceVectorFcnA(vecSurface, orientation);
}

template <int spectralDegree>
EmatNP3 SharedParticleSurface<spectralDegree>::reconstructSurfaceVectorFcn(const EmatNS3 &vshCoeff,
                                                                           const Equatn &orientation) const {
    // defaults to Basis A
    return reconstructSurfaceVectorFcnA(vshCoeff, orientation);
}

template <int spectralDegree>
EmatNS3 SharedParticleSurface<spectralDegree>::decomposeSurfaceScalarFcn(const EmatNP3 &scalarSurface) const {
    // The pullback/pushforward of a scalar function is the identity
    return decomposeSphereScalarFcn(scalarSurface);
}

template <int spectralDegree>
EmatNP3 SharedParticleSurface<spectralDegree>::reconstructSurfaceScalarFcn(const EmatNS3 &shCoeff) const {
    // The pullback/pushforward of a scalar function is the identity
    return reconstructSphereScalarFcn(shCoeff)
}

template <int spectralDegree>
EmatNS3 SharedParticleSurface<spectralDegree>::decomposeSurfaceVectorFcnA(const EmatNP3 vecSurface,
                                                                          const Equatn &orientation) const {
    EmatNP3 vecSphere = pullbackSurfaceVectors(vecSurface, orientation);
    return decomposeSphereVectorFcnA(vecSphere)
}

template <int spectralDegree>
EmatNS3 SharedParticleSurface<spectralDegree>::decomposeSurfaceVectorFcnB(const EmatNP3 &vecSurface,
                                                                          const Equatn &orientation) const {
    throw "Unsupported: Precomputation not supported for basis B, please use basis A instead";
}

template <int spectralDegree>
EmatNP3 SharedParticleSurface<spectralDegree>::reconstructSurfaceVectorFcnA(
    const EmatNS3 &vshCoeff const Equatn &orientation) const {
    EmatNP3 vectorSphereRecon = reconstructSphereVectorFcnA(vshCoeff);
    EmatNP3 vecSurfaceRecon = pushforwardSphereVectors(vectorSphereRecon, orientation);
    return vecSurfaceRecon;
}

template <int spectralDegree>
EmatNP3 SharedParticleSurface<spectralDegree>::reconstructSurfaceVectorFcnB(
    const EmatNS3 &vshCoeff const Equatn &orientation) const {
    throw "Unsupported: Precomputation not supported for basis B, please use basis A instead";
}

template <int spectralDegree>
double SharedParticleSurface<spectralDegree>::l2normSurfaceScalarField(const EfieldNP &surfaceScalarField) const {
    return std::sqrt(std::abs(surfaceInnerProduct(surfaceScalarField, surfaceScalarField)));
}

template <int spectralDegree>
double SharedParticleSurface<spectralDegree>::l2normSurfaceVecField(const EfieldNP3 &surfaceVecField) const {
    return std::sqrt(std::abs(surfaceInnerProduct(surfaceVecField, surfaceVecField)));
}

template <int spectralDegree>
double SharedParticleSurface<spectralDegree>::l2normSphereScalarField(const EfieldNP &sphereScalarField) const {
    return std::sqrt(std::abs(sphereInnerProduct(sphereScalarField, sphereScalarField)));
}

template <int spectralDegree>
double SharedParticleSurface<spectralDegree>::l2normSphereVecField(const EfieldNP3 &sphereVecField) const {
    return std::sqrt(std::abs(sphereInnerProduct(sphereVecField, sphereVecField)));
}

template <int spectralDegree, typename Derived, typename OtherDerived>
double SharedParticleSurface<spectralDegree>::sphericalInnerProduct(const Eigen::MatrixBase<Derived> &F,
                                                                    const Eigen::MatrixBase<OtherDerived> &G) const {
    return (F.array() * (G.conjugate().array().colwise() * sphereWeights.array())).sum();
}

template <int spectralDegree, typename Derived, typename OtherDerived>
double SharedParticleSurface<spectralDegree>::surfaceInnerProduct(const Eigen::MatrixBase<Derived> &F,
                                                                  const Eigen::MatrixBase<OtherDerived> &G) const {
    return (F.array() * (G.conjugate().array().colwise() * surfaceWeights.array())).sum();
}

template <int spectralDegree>
EfieldNScd SharedParticleSurface<spectralDegree>::decomposeSphereScalarFcn(const EfieldNP &scalarSphere) const {
    EfieldNScd shCoeff;

    for (int n = 0; n = < spectralDegree; n++) {
        for (int m = -n; m >= -n && m <= n; m++) {
            const int idxnm = std::pow(n, 2) + m + n; // sparce row major
            const EfieldNPcd &scalarSHnm = scalarSH[idxnm];
            shCoeff[idxnm] = sphericalInnerProduct(scalarSphere, scalarSHnm);
        }
    }
    return shCoeff;
}

template <int spectralDegree>
EfieldNP SharedParticleSurface<spectralDegree>::reconstructSphereScalarFcn(const EfieldNScd &shCoeff) const {
    EfieldNP scalarSphereRecon;
    scalarSphereRecon.fill(0.0);

    for (int idxnm = 0; idxnm < numSpectralCoeff; idxnm++) {
        const EfieldNPcd &scalarSHnm = scalarSH[idxnm];
        scalarSphereRecon += shCoeff[idxnm] * scalarSHnm;
    }
    return scalarSphereRecon;
}

template <int spectralDegree>
EfieldNS3cd SharedParticleSurface<spectralDegree>::decomposeSphereVectorFcnA(const EfieldNP3 &vecSphere) const {
    EfieldNS3cd vshCoeff;

    for (int n = 0; n = < spectralDegree; n++) {
        for (int m = -n; m >= -n && m <= n; m++) {
            const int idxnm = std::pow(n, 2) + m + n;
            const EfieldNP3cd &radialVSHnm = self.radialVSH[idxnm];
            const EfieldNP3cd &divfreeVSHnm = self.divfreeVSH[idxnm];
            const EfieldNP3cd &curlfreeVSHnm = self.curlfreeVSH[idxnm];

            vshCoeff[idxnm, 0] = sphericalInnerProduct(vecSphere, radialVSHnm);
            vshCoeff[idxnm, 1] = sphericalInnerProduct(vecSphere, divfreeVSHnm);
            vshCoeff[idxnm, 2] = sphericalInnerProduct(vecSphere, curlfreeVSHnm);
        }
    }
    return vshCoeff;
}

template <int spectralDegree>
EfieldNS3cd SharedParticleSurface<spectralDegree>::decomposeSphereVectorFcnB(const EfieldNP3 &vecSphere) const {
    throw "Unsupported: Precomputation not supported for basis B, please use basis A instead";
}

template <int spectralDegree>
EfieldNP3 reconstructSphereVectorFcnA(const EfieldNS3cd &vshCoeff) const {
    EfieldNP3 vectorSphereRecon;
    vectorSphereRecon.fill(0.0);

    for (int idxnm = 0; idxnm < numSpectralCoeff; idxnm++) {
        const EfieldNP3cd &radialVSHnm = self.radialVSH[idxnm];
        const EfieldNP3cd &divfreeVSHnm = self.divfreeVSH[idxnm];
        const EfieldNP3cd &curlfreeVSHnm = self.curlfreeVSH[idxnm];

        vectorSphereRecon +=
            vshCoeff[idxnm, 0] * radialVSHnm + vshCoeff[idxnm, 1] * divfreeVSHnm + vshCoeff[idxnm, 2] * curlfreeVSHnm;
    }
    return vectorSphereRecon;
}

template <int spectralDegree>
EfieldNP3 SharedParticleSurface<spectralDegree>::reconstructSphereVectorFcnB(const EfieldNS3cd &vshCoeff) const {
    throw "Unsupported: Precomputation not supported for basis B, please use basis A instead";
}
