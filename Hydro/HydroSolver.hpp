/**
 * @file HydroSolver.hpp
 * @author Bryce Palmer (palme200@msu.ed)
 * @brief Solve the hydrodynamic effect between rods using a Galerkin boundary integral method
 * @version 1.0
 * @date May 17, 2022
 *
 * See Eduardo Corona's An integral equation formulation for rigid bodies in Stokes
 * flow in three dimensions for details on the descrete boundary.
 * This will be superseeded by my own publication in following months.
 */
#ifndef HYDROSOLVER_HPP_
#define HYDROSOLVER_HPP_

#include "Particle/ParticleConfig.hpp"
#include "Util/EigenDef.hpp"
#include <complex>

// TODO
//  1. (done) Start by assuming that everything is double but then switch from double to std::complex<double> where
//   necessary
//  2. (done) Switch surface functions to be primarily void
//  3. (done) Replace np.cross with something else!
//  4. (done) All functions should use and alpha/beta to allow for Y := alpha Ax + beta Y

// Kronecker delts
int kdelta(int i, int j) { return i == j ? 1 : 0; }

Evec3 stokeslet(const Evec3 &rVec, const Evec3 &forceVec) {
    constexpr double Pi = M_PI;

    const double coeff = 1.0 / (8 * Pi);
    const double rNorm = std::sqrt((rVec.array() * rVec.array()).sum());
    const double rNormInvScaled = 1 / rNorm * coeff;
    const double rNormInv3Scaled = 1 / std::pow(rNorm, 3) * coeff;

    Evec3 velocityVec(0.0, 0.0, 0.0);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            velocityVec[i] += (kdelta(i, j) * rNormInvScaled + rNormInv3Scaled * rVec[i] * rVec[j]) * forceVec[j];
        }
    }
    return velocityVec;
}

Emat3 rstokeslet(const Evec3 &rVec) {
    constexpr double Pi = M_PI;

    const double coeff = 1.0 / (8 * Pi);
    const double rNorm = std::sqrt((rVec.array() * rVec.array()).sum());
    Emat3 rS;
    if (rNorm < 1e-8) {
        const double rNormInv2 = 1 / std::pow(rNorm, 2);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rS[i, j] = coeff * (rNormInv2 * rVec[i] * rVec[j] + kdelta(i, j));
            }
        }
    } else {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rS[i, j] = coeff * kdelta(i, j);
            }
        }
    }

    return rS;
}

Evec3 stresslet(const Evec3 &rVec, const Evec3 &forceVecTarget, const Evec3 &forceVecSource,
                const Evec3 &normalVecTarget, const Evec3 &normalVecSource) {
    constexpr double Pi = M_PI;

    const double coeff = -3.0 / (4 * Pi);
    const double rNorm = std::sqrt((rVec.array() * rVec.array()).sum());
    const double rNormInv5Scaled = 1 / std::pow(rNorm, 5) * coeff;

    Evec3 tractionVec(0.0, 0.0, 0.0);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                tractionVec[i] += rNormInv5Scaled * rVec[i] * rVec[j] * rVec[k] *
                                  (forceVecSource[j] * normalVecTarget[k] + forceVecTarget[k] * normalVecSource[j])
            }
        }
    }

    return tractionVec;
}

/**
 * @brief HydroSolver class
 * Containing all the functionality for computing and solving for the hydrodynamic effect between particles
 * using a Galerkin boundary integral problem. Linear problem solved via GMRES
 *
 * Make sure this is only created once per memory block!
 */
template <class Container>
class HydroSolver {

  private:
    const ParticleConfig &runConfig;             ///< system configuration
    const Container *const particleContainerPtr; ///< read-only!
    const int numPtcLocal;                       ///< local number of particles

    // indices, maps, etc, precomputed in constructor
    // Q: number of grid points per particle
    // S: number of spectral coefficients per particle
    Teuchos::RCP<const TCOMM> commRcp;
    Teuchos::RCP<TMAP> particleMapRcp;      ///< 1 dof per particle
    Teuchos::RCP<TMAP> particlePtsMapRcp;   ///< Q dof per particle
    Teuchos::RCP<TMAP> particleCoeffMapRcp; ///< S dof per particle
    Teuchos::RCP<TMAP> surfaceValuesMapRcp; ///< 3Q dof per particle
    Teuchos::RCP<TMAP> surfaceCoeffMapRcp;  ///< 3S dof per particle
    std::vector<int> particlePts;           ///< 1 dof per particle, stores Q
    std::vector<int> particleCoeffs;        ///< 1 dof per particle, stores S
    std::vector<int> particlePtsIndex;      ///< beginning of quadrature points per particle in particlePtsMap
    std::vector<int> particleCoeffIndex;    ///< beginning of spectral coefficients per particle in particleCoeffMap

    // these are precomputed/stored in constructor
    std::vector<double> totalExtForceTorque; ///< external force/torque specified in HydroTable, 6 dof per rod

    // We use precomputation to avoid the need to store these values directly
    // These values can be fetched from SharedParticleSurface for minimal memory use
    //
    // std::vector<double> surfaceWeights; ///< surface GL quadrature points, excludes the north and south pole
    // std::vector<double> surfaceNorms;   ///< surface normals, excludes the north and south pole
    // std::vector<double> surfacePoints;  ///< surface coordinates, excludes the north and south pole
    // std::vector<double> spherePoints;   ///< pullback of surface coordinates to sphere in ref config,
    //                                     ///<  excludes the north and south pole

    // mutable data structures
    // remember, mutable data structures can be changed by const methods!
    mutable std::vector<double> surfaceDensityGridVals;             ///< 3Q points per rod
    mutable std::vector<double> surfaceVelocityGridVals;            ///< 3Q points per rod
    mutable std::vector<std::complex<double>> surfaceDensityCoeff;  ///< 3S points per rod
    mutable std::vector<std::complex<double>> surfaceVelocityCoeff; ///< 3S points per rod

  public:
    /**
     * @brief Construct a new HydroSolver object
     *
     * precompute() should be called after this constructor
     */
    HydroSolver() = default;

    /**
     * @brief Construct a new HydroSolver object
     *
     * This constructor calls precompute() internally
     * @param configFile a yaml file for ParticleConfig
     * @param posFile initial configuration. use empty string ("") for no such file
     * @param argc command line argument
     * @param argv command line argument
     */
    HydroSolver(const std::string &name_, const std::string &particleShape_, int argc, char **argv);

    // default destructor
    ~HydroSolver() = default;

    // forbid copy
    HydroSolver(const HydroSolver &) = delete;
    HydroSolver &operator=(const HydroSolver &) = delete;

    //
    // These functions are required since we inherit from Tpetra::Operator
    //
    Teuchos::RCP<const TMAP> getDomainMap() const { return surfaceCoeffMapRcp; }

    Teuchos::RCP<const TMAP> getRangeMap() const { return surfaceCoeffMapRcp; }

    bool hasTransposeApply() const { return false; }

    // Compute Y := alpha Op X + beta Y.
    void apply(const TMV &X, TMV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
               scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one(),
               scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero()) const;

    // // get read-only reference
    // const std::vector<double> &getSurfaceWeights() const { return surfaceWeights; }
    // const std::vector<double> &getSurfaceNorms() const { return surfaceNorms; }
    // const std::vector<double> &getSurfacePoints() const { return surfacePoints; }
    // const std::vector<double> &getSpherePoints() const { return spherePoints; }

    void postProcessAndStoreResults(const std::vector<std::complex<double>> &surfaceDensityCoeff,
                                    const std::vector<double> &surfaceDensityGridVals);

    // Y := alpha A x + beta Y
    void measureAx(const std::complex<double> *unknownSurfaceDensityCoeff, double *unknownSurfaceDensityGridVals,
                   std::complex<double> *AxCoeff, double *AxGridVals, const double alpha = 1.0,
                   const double beta = 0.0) const {
        const auto &particleContainer = *containerPtr;

        // step 1: convert knownSurfaceDensityCoeff from spectral to grid
#pragma omp parallel for
        for (int pidx = 0; pidx < numPtcLocal; pidx++) {
            // particle properties
            const auto &ptc = particleContainer[pidx];
            const auto surface = ptc.getSurface();

            // reconstruct surfaceDensity from surfaceDensityCoeff
            const int coeffidx = self.particleCoeffIndex[pidx];
            const int pointidx = self.particlePtsIndex[pidx];

            const std::complex<double> *coeffVSHPtr = &unknownSurfaceDensityCoeff[3 * coeffidx];
            double *surfaceVectorPtr = &unknownSurfaceDensityGridVals[3 * pointidx];
            surface.reconstructSurfaceVectorFcn(coeffVSHPtr, surfaceVectorPtr);
        }

        // step 2: compute the operations on the grid values
        calcAx(unknownSurfaceDensityGridVals, AxGridVals, alpha, beta);

        // step 3: convert from grid to spectral
        for (int pidx = 0; pidx < numPtcLocal; pidx++) {
            // particle properties
            const auto &ptc = particleContainer[pidx];
            const auto surface = ptc.getSurface();

            // convert the grid values to spectral coeff
            const int pointidx = self.particlePtsIndex[pidx];
            const int coeffidx = self.particleCoeffIndex[pidx];

            const double *surfaceVectorPtr = &AxGridVals[3 * pointidx];
            std::complex<double> *coeffVSHPtr = &AxCoeff[3 * coeffidx];
            surface.decomposeSurfaceVectorFcn(surfaceVectorPtr, coeffVSHPtr);
        }
    }

    // Y = alpha*A*x + beta*Y
    void measureb(const std::complex<double> *knownSurfaceDensityCoeff, double *unknownSurfaceDensityGridVals,
                  std::complex<double> *bCoeff, double *bGridVals, const double alpha = 1.0,
                  const double beta = 0.0) const {
        // step 1: convert knownSurfaceDensityCoeff from spectral to grid
        for (int pidx = 0; pidx < numPtcLocal; pidx++) {
            // particle properties
            const auto &ptc = particleContainer[pidx];
            const auto surface = ptc.getSurface();

            // reconstruct surfaceDensity from surfaceDensityCoeff
            const int pointidx = self.particlePtsIndex[pidx];
            const int coeffidx = self.particleCoeffIndex[pidx];

            const std::complex<double> *coeffVSHPtr = &knownSurfaceDensityCoeff[3 * coeffidx];
            double *surfaceVectorPtr = &knownSurfaceDensityGridVals[3 * pointidx];
            surface.reconstructSurfaceVectorFcn(coeffVSHPtr, surfaceVectorPtr)
        }

        // step 2: compute the operations on the grid values
        calcb(self.knownSurfaceDensityGridVals, bGridVals, alpha, beta);

        // step 3: convert from grid to spectral
        for (int pidx = 0; pidx < numPtcLocal; pidx++) {
            // particle properties
            const auto &ptc = particleContainer[pidx];
            const auto surface = ptc.getSurface();

            // convert the grid values to spectral coeff
            const int pointidx = self.particlePtsIndex[pidx];
            const int coeffidx = self.particleCoeffIndex[pidx];

            const double *surfaceVectorPtr = &bGridVals[3 * pointidx];
            std::complex<double> *coeffVSHPtr = &bCoeff[3 * coeffidx];
            surface.decomposeSurfaceVectorFcn(surfaceVectorPtr, coeffVSHPtr);
        }
    }

    // Y = alpha*A*x + beta*Y
    void measureBx(const double *totalForceTorque, std::complex<double> *BxCoeff, double *BxGridVals,
                   double alpha = 1.0, double beta = 1.0) const {
        // step 1: compute Bx from the total force and torque
        calcBx(totalForceTorque, BxGridVals, alpha, beta);

        // step 2: convert from grid to spectral
        for (int pidx = 0; pidx < numPtcLocal; pidx++) {
            // particle properties
            const auto &ptc = particleContainer[pidx];
            const auto surface = ptc.getSurface();

            // measure densityFromInducedTotalForceAndTorque using VSH
            const int pointidx = self.particlePtsIndex[pidx];
            const int coeffidx = self.particleCoeffIndex[pidx];

            const double *surfaceVectorPtr = &BxGridVals[3 * pointidx];
            std::complex<double> *coeffVSHPtr = &BxCoeff[3 * coeffidx];
            surface.decomposeSurfaceVectorFcn(surfaceVectorPtr, coeffVSHPtr);
        }
    }

    // Y = alpha*A*x + beta*Y
    void calcAx(const double *unknownSurfaceDensityGridVals, double *AxGridVals, const double alpha = 1.0,
                const double beta = 0.0) const {
        calcLx(unknownSurfaceDensityGridVals, AxGridVals, alpha, beta);
        calcJx(unknownSurfaceDensityGridVals, AxGridVals, alpha, 1.0);
    }

    // Y = alpha*A*x + beta*Y
    void calcb(const double *knownSurfaceDensityGridVals, const double alpha = 1.0, const double beta = 0.0) const {
        return calcJx(knownSurfaceDensityGridVals, -alpha, beta);
    }

    // Y = alpha*A*x + beta*Y
    void calcBx(const double *totalForceTorque, double *BxGridVals, const double alpha = 1.0,
                const double beta = 0.0) const {
        for (int pidx = 0; pidx < numPtcLocal; pidx++) {
            // particle properties
            const auto &ptc = particleContainer[pidx];
            const auto surface = ptc.getSurface();
            const int numGridPts = surface.getNumGridPts();
            const Evec3 centroid = surface.getCentroidCurrentConfig(ptc.pos);
            const Emat3 invMomentOfInertiaTensor = surface.getInvMomentOfInertiaTensorCurrentConfig();
            const int invSurfaceArea = 1.0 / surface.getSurfaceArea();

            // calculate the total force and torque induced by surfaceDensity
            const Evec3 totalForce = ECmap3(totalForceTorque + 6 * pidx);
            const Evec3 totalTorque = ECmap3(totalForceTorque + 6 * pidx + 3);

            // distribute the total force and torque to the entire surface
            const Evec3 tauInvT = invMomentOfInertiaTensor * totalTorque;
            const int pointidx = self.particlePtsIndex[pidx];
            for (int j = 0; j < numGridPts; j++) {
                const Evec3 gridPoint = surface.getGridPointCurrentConfig(ptc.pos, j);
                const int idx = 3 * (pointidx + j);
                BxGridVals[idx + 0] = BxGridVals[idx + 0] * beta + alpha * totalForce[0] * invSurfaceArea;
                BxGridVals[idx + 1] = BxGridVals[idx + 1] * beta + alpha * totalForce[1] * invSurfaceArea;
                BxGridVals[idx + 2] = BxGridVals[idx + 2] * beta + alpha * totalForce[2] * invSurfaceArea;
                BxGridVals[idx] = BxGridVals[idx] * beta + alpha * (tauInvT[1] * (gridPoint[2] - centroid[2]) -
                                                                    tauInvT[2] * (gridPoint[1] - centroid[1]));
                BxGridVals[idx] = BxGridVals[idx] * beta + alpha * (tauInvT[2] * (gridPoint[0] - centroid[0]) -
                                                                    tauInvT[0] * (gridPoint[2] - centroid[2]));
                BxGridVals[idx] = BxGridVals[idx] * beta + alpha * (tauInvT[0] * (gridPoint[1] - centroid[1]) -
                                                                    tauInvT[1] * (gridPoint[0] - centroid[0]));
            }
        }
    }

    // Y = alpha*A*x + beta*Y
    void calcLx(const double *surfaceDensityGridVals, double *LxGridVals, const double alpha = 1.0,
                const double beta = 0.0) const {
        for (int pidx = 0; pidx < numPtcLocal; pidx++) {
            // particle properties
            const auto &ptc = particleContainer[pidx];
            const auto surface = ptc.getSurface();
            const int numGridPts = surface.getNumGridPts();
            const Evec3 centroid = surface.getCentroidCurrentConfig(ptc.pos);
            const Emat3 invMomentOfInertiaTensor = surface.getInvMomentOfInertiaTensorCurrentConfig();
            const int invSurfaceArea = 1.0 / surface.getSurfaceArea();

            // calculate the total force and torque induced by surfaceDensityGridVals
            Evec3 inducedTotalForce(0.0, 0.0, 0.0);
            Evec3 inducedTotalTorque(0.0, 0.0, 0.0);
            const int ptsidx = self.particlePtsIndex[pidx];
            for (int j = 0; j < numGridPts; j++) {
                const Evec3 gridPoint = surface.getGridPointCurrentConfig(ptc.pos, j);

                inducedTotalForce[0] += surfaceDensityGridVals[3 * (ptsidx + j) + 0] * surface.getGridWeight(j);
                inducedTotalForce[1] += surfaceDensityGridVals[3 * (ptsidx + j) + 1] * surface.getGridWeight(j);
                inducedTotalForce[2] += surfaceDensityGridVals[3 * (ptsidx + j) + 2] * surface.getGridWeight(j);
                inducedTotalTorque[0] += ((gridPoint[1] - centroid[1]) * surfaceDensityGridVals[3 * (ptsidx + j) + 2] -
                                          (gridPoint[2] - centroid[2]) * surfaceDensityGridVals[3 * (ptsidx + j) + 1]) *
                                         surface.getGridWeight(j);
                inducedTotalTorque[1] += ((gridPoint[2] - centroid[2]) * surfaceDensityGridVals[3 * (ptsidx + j) + 0] -
                                          (gridPoint[0] - centroid[0]) * surfaceDensityGridVals[3 * (ptsidx + j) + 2]) *
                                         surface.getGridWeight(j);
                inducedTotalTorque[2] += ((gridPoint[0] - centroid[0]) * surfaceDensityGridVals[3 * (ptsidx + j) + 1] -
                                          (gridPoint[1] - centroid[1]) * surfaceDensityGridVals[3 * (ptsidx + j) + 0]) *
                                         surface.getGridWeight(j);
            }

            // distribute the total force and torque to the entire surface
            const Evec3 inducedTotalTorque = invMomentOfInertiaTensor * inducedTotalTorque;
            for (int j = 0; j < numGridPts; j++) {
                const Evec3 gridPoint = surface.getGridPointCurrentConfig(ptc.pos, j);

                const int idx = 3 * (ptsidx + j);
                LxGridVals[idx + 0] = beta * LxGridVals[idx + 0] + alpha * inducedTotalForce[0] * invSurfaceArea;
                LxGridVals[idx + 1] = beta * LxGridVals[idx + 1] + alpha * inducedTotalForce[1] * invSurfaceArea;
                LxGridVals[idx + 2] = beta * LxGridVals[idx + 2] + alpha * inducedTotalForce[2] * invSurfaceArea;
                LxGridVals[idx + 0] =
                    beta * LxGridVals[idx + 0] + alpha * (inducedTotalTorque[1] * (gridPoint[2] - centroid[2]) -
                                                          inducedTotalTorque[2] * (gridPoint[1] - centroid[1]));
                LxGridVals[idx + 1] =
                    beta * LxGridVals[idx + 1] + alpha * (inducedTotalTorque[2] * (gridPoint[0] - centroid[0]) -
                                                          inducedTotalTorque[0] * (gridPoint[2] - centroid[2]));
                LxGridVals[idx + 2] =
                    beta * LxGridVals[idx + 2] + alpha * (inducedTotalTorque[0] * (gridPoint[1] - centroid[1]) -
                                                          inducedTotalTorque[1] * (gridPoint[0] - centroid[0]));
            }
        }
    }

    // Y = alpha*A*x + beta*Y
    void calcJx(const double *surfaceDensityGridVals, double *JxGridVals, const double alpha = 1.0,
                const double beta = 0.0) const {

        for (int pidxTarget = 0; pidxTarget < numPtcLocal; pidxTarget++) {
            // particle properties
            const auto &ptcTarget = particleContainer[pidx];
            const auto surfaceTarget = ptcTarget.getSurface();
            const int numGridPtsTarget = surface.getNumGridPts();
            const int ptsidxTarget = self.particlePtsIndex[pidxTarget];

            double *JTarget = &JxGridVals[3 * ptsidxTarget];
            for (int i = 0; i < numGridPtsTarget; i++) {
                const Evec3 gridPointTargeti = layerTarget.getGridPointCurrentConfig(ptcTarget.pos, i);
                const Evec3 gridNormTargeti = layerTarget.getGridNormCurrentConfig(i);
                const Evec3 gridDensityTargeti = ECmap3(surfaceDensityGridVals + 3 * (ptsidxTarget + i));

                for (int pidxSource = 0; pidxSource < numPtcLocal; pidxSource++) {
                    const bool selfInteraction = pidxTarget == pidxSource ? true : false;

                    // particle properties
                    const auto &ptcSource = particleContainer[pidx];
                    const auto surfaceSource = ptcTarget.getSurface();
                    const int numGridPtsSource = surface.getNumGridPts();
                    const int ptsidxSource = self.particlePtsIndex[pidxSource];

                    for (int j = 0; j < numGridPtsSource; j++) {
                        // singular subtraction allows us to skip the singular self-interaction component
                        if (selfInteraction) {
                            if (i == j) {
                                continue;
                            }
                        }

                        const Evec3 gridPointSourcej = layerSource.getGridPointCurrentConfig(ptcSource.pos, j);
                        const Evec3 gridNormSourcej = layerSource.getGridNormCurrentConfig(j);
                        const Evec3 gridDensitySourcej = ECmap3(surfaceDensityGridVals + 3 * (ptsidxSource + j));

                        const Evec3 rVec = gridPointTargeti - gridPointSourcej;
                        const Evec3 tractionVec = self.stresslet(rVec, gridDensityTargeti, gridDensitySourcej,
                                                                 gridNormTargeti, gridNormSourcej);
                        Emap3(JTarget + 3 * i) =
                            beta * Emap3(JTarget + 3 * i) + alpha * tractionVec * layerSource.getGridWeight(j);
                    }
                }
            }
        }
    }
};

#endif