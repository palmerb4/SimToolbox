#ifndef HYDROPARTICLEOPERATOR_HPP_
#define HYDROPARTICLEOPERATOR_HPP_

#include "SharedParticleSurface.hpp"
#include "STKFMM/STKFMM.hpp"

#include "Particle/ParticleConfig.hpp"
#include "Trilinos/Preconditioner.hpp"
#include "Trilinos/TpetraUtil.hpp"
#include "Util/EigenDef.hpp"


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

template <class Container>
class HydroParticleOperator : public TOP {
  private:
    const ParticleConfig &runConfig;             ///< system configuration
    const Container const *particleContainerPtr; ///< a container for the particles. read only!
    int numPtcLocal;                             ///< local number of particles

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
    std::vector<double> totalExtForceTorque; ///< external force/torque specified in HydroTable, 6 dof per particle

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
    mutable std::vector<double> particleVelOmega;                         ///< 6 points per particle
    mutable std::vector<double> knownSurfaceDensityGridVals;              ///< 3Q points per particle
    mutable std::vector<double> unknownSurfaceDensityGridVals;            ///< 3Q points per particle
    mutable std::vector<double> AxGridVals;                               ///< 3Q points per particle
    mutable std::vector<double> BxGridVals;                               ///< 3Q points per particle
    mutable std::vector<double> bGridVals;                                ///< 3Q points per particle
    mutable std::vector<std::complex<double>> knownSurfaceDensityCoeff;   ///< 3S points per particle
    mutable std::vector<std::complex<double>> unknownSurfaceDensityCoeff; ///< 3S points per particle
    mutable std::vector<std::complex<double>> AxCoeff;                    ///< 3S points per particle
    mutable std::vector<std::complex<double>> BxCoeff;                    ///< 3S points per particle
    mutable std::vector<std::complex<double>> bCoeff;                     ///< 3S points per particle

    void setupDOF();

    void measureBx(const double *totalForceTorquePtr, std::complex<double> *BxCoeffPtr, double *BxGridValsPtr,
                   double alpha = 1.0, double beta = 1.0) const;

    void measureLx(const std::complex<double> *surfaceDensityCoeffPtr, std::complex<double> *LxCoeffPtr,
                   const double alpha = 1.0, const double beta = 0.0) const;

    void calcAx(const double *unknownSurfaceDensityGridValsPtr, double *AxGridValsPtr, const double alpha = 1.0,
                const double beta = 0.0) const;

    void calcb(const double *knownSurfaceDensityGridValsPtr, const double alpha = 1.0, const double beta = 0.0) const;

    void calcBx(const double *totalForceTorquePtr, double *BxGridValsPtr, const double alpha = 1.0,
                const double beta = 0.0) const;

    void calcLx(const double *surfaceDensityGridValsPtr, double *LxGridValsPtr, const double alpha = 1.0,
                const double beta = 0.0) const;

    void calcJx(const double *surfaceDensityGridValsPtr, double *JxGridValsPtr, const double alpha = 1.0,
                const double beta = 0.0) const;

  public:
    /************************************************
     * Interface required by TOP
     *
     *
     *
     ***********************************************/

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
    HydroParticleOperator(const Container *const particleContainerPtr, const int nParticleLocal,
                          const SylinderConfig &runConfig);

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

    /***********************************************
     * Interface for hydro calculation
     *
     *
     *
     ***********************************************/

    void cacheResults(Container &particleContainer) const;

    void getSurfaceDensityCoeffFromParticleContainer(std::complex<double> *surfaceDensityCoeffPtr) const;

    void getUserInputSurfaceDensityCoeff(std::complex<double> *userInputSurfaceDensityCoeff, const double alpha = 1.0,
                                         const double beta = 0.0) const;


    void calcParticleVelOmega(double *particleVelOmega) const;

    void measureb(const std::complex<double> *knownSurfaceDensityCoeffPtr, double *unknownSurfaceDensityGridValsPtr,
                  std::complex<double> *bCoeffPtr, double *bGridValsPtr, const double alpha = 1.0,
                  const double beta = 0.0) const;

    void measureAx(const std::complex<double> *unknownSurfaceDensityCoeffPtr, double *unknownSurfaceDensityGridValsPtr,
                   std::complex<double> *AxCoeffPtr, double *AxGridValsPtr, const double alpha = 1.0,
                   const double beta = 0.0) const;

    void getSurfaceDensityCoeffFromParticleContainerWrapper(TV &surfaceDensityCoeffPtr) const;

    void getUserInputSurfaceDensityCoeffWrapper(TV &userInputSurfaceDensityCoeff, const double alpha = 1.0,
                                                const double beta = 0.0) const;

    void calcParticleVelOmegaWrapper(TV &particleVelOmega) const;

    void measurebWrapper(const TV &knownSurfaceDensityCoeffVec, TV &bCoeffVec,
                         const bool withUserInputExternalForceTorque, const bool withUserInputSurfaceDensity) const;

    void measureAxWrapper(const TV &unknownSurfaceDensityCoeffVec, TV &AxCoeffVec) const;

    // get read-only references
    const std::vector<double> &getParticleVelOmega() const { return particleVelOmega; }
    const std::vector<double> &getSurfaceDensityGridVals() const { return knownSurfaceDensityGridVals; }
    const std::vector<double> &getUnknownSurfaceDensityGridVals() const { return unknownSurfaceDensityGridVals; }
    const std::vector<std::complex<double>> &getKnownSurfaceDensityCoeff() const { return knownSurfaceDensityCoeff; }
    const std::vector<std::complex<double>> &getUnknownSurfaceDensityCoeff() const {
        return unknownSurfaceDensityCoeff;
    }
};

// Include the HydroParticleOperator implimentation
#include "HydroParticleOperator.tpp"

#endif