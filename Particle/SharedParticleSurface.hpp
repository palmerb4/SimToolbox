/**
 * @file SharedParticleSurface.hpp
 * @author Bryce Palmer (palme200@msu.ed)
 * @brief Precomputed descrete surface for a single particle
 * @version 1.0
 * @date May 16, 2022
 *
 */
#ifndef SHAREDPARTICLESURFACE_HPP_
#define SHAREDPARTICLESURFACE_HPP_

#include "Util/EigenDef.hpp"

#include <complex>

/**
 * @brief SharedParticleSurface class
 * Containing precomputed particle surface information shared among multiple particles
 *  as well as optimized functions using this information
 *
 */
template <int spectralDegree>
class SharedParticleSurface {

  private:
    std::string name;
    std::shared_ptr<const ParticleShapes> particleShapesPtr;
    Evec3 north;
    std::string prescribedSurfaceDensityFile;

    // compile time variables
    // Assume Guass Legandre quadrature (without poles)
    constexpr int numGridPts = (spectralDegree + 1) * (2 * spectralDegree + 2);
    constexpr int numSpectralCoeff = std::pow(spectralDegree + 1, 2);

    // useful Eigen functionality
    // scalar fields
    using EfieldNP = Eigen::Matrix<double, numGridPts, 1, Eigen::DontAlign>;
    using EfieldmapNP = Eigen::Map<EfieldNP, Eigen::Unaligned>;
    using EfieldNPcd = Eigen::Matrix<std::complex<double>, numGridPts, 1, Eigen::DontAlign>;
    using EfieldmapNPcd = Eigen::Map<EfieldNPcd, Eigen::Unaligned>;
    using EfieldNScd = Eigen::Matrix<std::complex<double>, numSpectralCoeff, 1, Eigen::DontAlign>;
    using EfieldmapNScd = Eigen::Map<EfieldNScd, Eigen::Unaligned>;

    // vector fields
    using EfieldNP3 = Eigen::Matrix<double, numGridPts, 3, Eigen::DontAlign>;
    using EfieldmapNP3 = Eigen::Map<EfieldNP, Eigen::Unaligned>;
    using EfieldNP3cd = Eigen::Matrix<std::complex<double>, numGridPts, 3, Eigen::DontAlign>;
    using EfieldmapNP3cd = Eigen::Map<EfieldNPcd, Eigen::Unaligned>;
    using EfieldNS3cd = Eigen::Matrix<std::complex<double>, numSpectralCoeff, 3, Eigen::DontAlign>;
    using EfieldmapNS3cd = Eigen::Map<EfieldNS3cd, Eigen::Unaligned>;

    // NP3 x NP3 matrix
    using EmatNP3 = Eigen::Matrix<double, numGridPts * 3, numGridPts * 3, Eigen::DontAlign>;

    // Precomputed quantities in the reference configeration (with poles)
    EfieldNP thetasA;       //<<< azimuthal angles to each quadrature point with z-axis pole
    EfieldNP phisA;         //<<< polar angles to each quadrature point with z-axis pole
    EfieldNP thetasB;       //<<< azimuthal angles to each quadrature point with x-axis pole
    EfieldNP phisB;         //<<< polar angles to quadrature point with x-axis pole
    EfieldNP sphereWeights; //<<< Guass Lagandre quadrature weights at each quadrature point on the sphere
    EfieldNP gridWeights;   //<<< Modified Guass Lagandre quadrature weights at each quadrature point on the surface
    EfieldNP spherePointsRefConfig;          //<<< location of each quadrature point on the sphere in the ref config
    EfieldNP3 gridPointsRefConfig;           //<<< location of each quadrature point on the surface in the ref config
    EfieldNP3 gridNormsRefConfig;            //<<< surface normal vectors in the ref config
    EmatNP3 pullbackMatrix;                  //<<< maps flattened, row-major surface vector fields to
                                             //    flattened, row-major sphere vector fields
    EmatNP3 pushforwardMatrix;               //<<< maps flattened, row-major sphere vector fields to
                                             //    flattened, row-major surface vector fields
    Emat3 invMomentOfInertiaTensorRefConfig; //<<< inverse moment of inertia tensor
    Evec3 centroidRefConfig;                 //<<< centroid of the surface in the ref config (need not be zero!)
    double surfaceArea;                      //<<< surface area

    /*Precomputed spectral information
     * Note: Each grid point contains numSpectralCoeff spectral values.
     *  Eigen does not yet support tensors, so we must store data in an array
     *  Data is stored using an std::array of Eigen matrices
     *  Index using idxnm to fetch all grid point data for a given n,m pair
     */
    EfieldNS3cd prescribedSurfaceDensityCoeff;
    EfieldNPcd scalarSH[numSpectralCoeff];
    EfieldNP3cd radialVSH[numSpectralCoeff];
    EfieldNP3cd divfreeVSH[numSpectralCoeff];
    EfieldNP3cd curlfreeVSH[numSpectralCoeff];

    /**
     * @brief Precompute all temporally invariant quantities
     */
    void precompute();

    /**
     * @brief Precompute and store the locations and weights of the Gauss-Legendre quadrature for the sphere
     *        rotated such that true north [0.0, 0.0, 1.0] is rotated to this->north
     * 
     *  The quadrature has
     *  (p+1)(2p+2) + north/south pole = 2p^2+4p+4 total point with pole
     *
     * This function precomputes the following member functions
     * @param thetasA, azimuthal angles with z-axis pole at each quadrature point
     * @param phisA, polar angles with z-axis pole at each quadrature point
     * @param thetasB, azimuthal angles with x-axis pole at each quadrature point
     * @param phisB, polar angles with x-axis pole at each quadrature point
     * @param spherePointsRefConfig, location of each quadrature point on the sphere
     * @param sphereWeights, quadrature weights at each quadrature point on the sphere
     */
    void precomputeGuassLegandreQuadrature();

    /**
     * @brief Precompute and store all constant surface properties
     *
     * This function precomputes the following member functions
     * @param gridNormsRefConfig
     * @param gridWeights
     * @param gridPointsRefConfig
     * @param pullbackMatrix
     * @param pushforwardMatrix
     */
    void precomputeSurfaceProperties();

    /**
     * @brief Precompute and store the spherical and vector spherical hamonics
     *        at each grid point in the reference config.
     *
     * This function precomputes the following member functions
     * @param scalarSH
     * @param radialVSH
     * @param divfreeVSH
     * @param curlfreeVSHnm
     */
    void precomputeSpectralHarmonics();

    /**
     * @brief Store the user defined surface density from the given csv file
     *
     * This function precomputes the following member functions
     * @param prescribedSurfaceDensityCoeff The vector spherical harmonic coefficients
     *           of the decomposed surface density. The coefficients are stored such that
     *          (idx 0: radial, idx 1: divFree, idx 2: curlFree)
     */
    void storePrescribedSurfaceDensityCoeff() {
        // read in the data
        rapidcsv::Document doc(prescribedSurfaceDensityFile, rapidcsv::LabelParams(0, -1));
        std::vector<std::string> columnNames = doc.GetColumnNames();
        for (auto &c : columnNames) {
            std::cout << c << std::endl;
        }
        const std::vector<int> nData = doc.GetColumn<int>("n");
        const std::vector<int> mData = doc.GetColumn<int>("slipVelocity");
        const std::vector<std::complex<double>> vshCoeffRadialAData = doc.GetColumn<double>("vshCoeffRadialA");
        const std::vector<std::complex<double>> vshCoeffDivFreeAData = doc.GetColumn<double>("vshCoeffDivFreeA");
        const std::vector<std::complex<double>> vshCoeffCurlFreeAData = doc.GetColumn<double>("vshCoeffCurlFreeA");

        // store the data
        const int numRows = vshCoeffCurlFreeA.size();
        for (int i = 0; i < numRows; i++) {
          const int idxnm = nData[i]*nData[i] + mData[i] + nData[i];
          prescribedSurfaceDensityCoeff[idxnm, 0] = vshCoeffRadialAData[i];
          prescribedSurfaceDensityCoeff[idxnm, 1] = vshCoeffDivFreeAData[i];
          prescribedSurfaceDensityCoeff[idxnm, 2] = vshCoeffCurlFreeAData[i];
        }
    }

    /**
     * @brief Compute the spectral decomposition of a surface vector evaluated at each grid point using basis A
     *
     * @param vecSurface A surface vector evaluated at each grid point
     *
     * @return vshCoeff The vector spherical harmonic coefficients
     *          (idx 0: radial, idx 1: divFree, idx 2: curlFree)
     */
    EfieldNS3 decomposeSurfaceVectorFcnA(const EfieldNP3 &vecSurface, const Equatn &orientation) const;

    /**
     * @brief Compute the spectral decomposition of a surface vector evaluated at each grid point using basis B
     *
     * @param vecSurface A surface vector evaluated at each grid point
     *
     * @return vshCoeff The vector spherical harmonic coefficients
     *          (idx 0: radial, idx 1: divFree, idx 2: curlFree)
     */
    EfieldNS3 decomposeSurfaceVectorFcnB(const EfieldNP3 &vecSurface, const Equatn &orientation) const;

    /**
     * @brief Compute the reconstruction of a surface vector based on its vector spherical hamonic coefficients using
     * basis A
     *
     * @param vshCoeff The vector spherical harmonic coefficients
     *          (idx 0: radial, idx 1: divFree, idx 2: curlFree)
     *
     * @return The reconstructed surface vector evaluated at each grid point
     */
    EfieldNP3 reconstructSurfaceVectorFcnA(const EfieldNS3 &vshCoeff const Equatn &orientation) const;

    /**
     * @brief Compute the reconstruction of a surface vector based on its vector spherical hamonic coefficients using
     * basis B
     *
     * @param vshCoeff The vector spherical harmonic coefficients
     *          (idx 0: radial, idx 1: divFree, idx 2: curlFree)
     *
     * @return The reconstructed surface vector evaluated at each grid point
     */
    EfieldNP3 reconstructSurfaceVectorFcnB(const EfieldNS3 &vshCoeff const Equatn &orientation) const;

    /**
     * @brief Compute the spectral decomposition of a sphere vector evaluated at each grid point using basis A
     *
     * @param vecSphere A sphere vector evaluated at each grid point
     *
     * @return The vector spherical harmonic coefficients of the decomposition
     */
    EfieldNS3cd decomposeSphereVectorFcnA(const EfieldNP3 &vecSphere) const;

    /**
     * @brief Compute the spectral decomposition of a sphere vector evaluated at each grid point using basis B
     *
     * @param vecSphere A sphere vector evaluated at each grid point
     *
     * @return The vector spherical harmonic coefficients of the decomposition
     */
    EfieldNS3cd decomposeSphereVectorFcnB(const EfieldNP3 &vecSphere) const;

    /**
     * @brief Compute the reconstruction of a sphere vector based on its vector spherical hamonic coefficients using
     * basis A
     *
     * @param vshCoeff The vector spherical harmonic coefficients
     *          (idx 0: radial, idx 1: divFree, idx 2: curlFree)
     *
     * @return The reconstructed sphere vector evaluated at each grid point
     */
    EfieldNP3 reconstructSphereVectorFcnA(const EfieldNS3cd &vshCoeff) const;

    /**
     * @brief Compute the reconstruction of a sphere vector based on its vector spherical hamonic coefficients using
     * basis B
     *
     * @param vshCoeff The vector spherical harmonic coefficients
     *          (idx 0: radial, idx 1: divFree, idx 2: curlFree)
     *
     * @return The reconstructed sphere vector evaluated at each grid point
     */
    EfieldNP3 reconstructSphereVectorFcnB(const EfieldNS3cd &vshCoeff) const;

  public:
    /**
     * @brief Construct a new SharedParticleSurface object
     *
     * precompute() should be called after this constructor
     */
    SharedParticleSurface() = default;

    /**
     * @brief Construct a new SharedParticleSurface object
     *
     * This constructor calls precompute() internally
     * @param configFile a yaml file for ParticleConfig
     * @param posFile initial configuration. use empty string ("") for no such file
     * @param argc command line argument
     * @param argv command line argument
     */
    SharedParticleSurface(const std::string &name, const ParticleShapes &particleShape, const Evec3 north = {0.0, 0.0, 1.0}, const std::string prescribedSurfaceDensityFile = "");

    ~SharedParticleSurface() = default;

    // forbid copy
    SharedParticleSurface(const SharedParticleSurface &) = delete;
    SharedParticleSurface &operator=(const SharedParticleSurface &) = delete;

    /**
     * @brief Get the number of grid points on the surface
     *
     * @return Number of grid points on the surface
     */
    int getNumGridPts() const;

    /**
     * @brief Get the number of spectral coefficients used in the discretization of the surface
     *  This class must return a const expression, as it is used to initialize various arrays
     *
     * @return Spectral coefficients used in the discretization of the surface
     */
    int getNumSpectralCoeff() const;

    /**
     * @brief Get the gridPoint of the surface in the current configuration
     *
     * @param coordBase The coordinate of the particle's core, not necessarily its centroid
     * @param idx The index of the grid point to fetch
     *
     * @return The gridPoint of the surface in the current configuration
     */
    Evec3 getGridPointCurrentConfig(const Evec3 &coordBase, const Equatn &orientation, const int &idx) const;

    /**
     * @brief Get the normalized normal vector of the surface in the current configuration
     *
     * @param coordBase The coordinate of the particle's core, not necessarily its centroid
     * @param idx The index of the grid point to fetch
     *
     * @return The normalized normal vector of the surface in the current configuration
     */
    Evec3 getGridNormCurrentConfig(const Equatn &orientation, const int &idx) const;

    /**
     * @brief Get the moment of inertia tensor of the surface in the current configuration
     *
     * @return The moment of inertia tensor of the surface in the current configuration
     */
    Emat3 getInvMomentOfInertiaTensorCurrentConfig(const Equatn &orientation) const;

    /**
     * @brief Get the grid weight corresponding to idx'th quadrature point
     *
     * @param idx The index of the grid point to fetch
     *
     * @return The idx'th grid weight
     */
    double getGridWeight(const int &idx) const;

    /**
     * @brief Get the surface area of the object
     *
     * @return The surface area of the object
     */
    double getSurfaceArea() const;

    /**
     * @brief Rotate every element of a surface vector from the current config to the ref config
     *
     * @param vecSurfaceCurrentConfig Surface vector in the current config
     * @param orientation Quaternion specifying the surface's orientation about its core.
     *
     * @return The surface vector in the ref config
     */
    EfieldNP3 rotateSurfaceVecCurrentConfigToRefConfig(const EfieldNP3 &vecSurfaceCurrentConfig,
                                                       const Equatn &orientation) const;

    /**
     * @brief Rotate every element of a surface vector from the ref config to the current config
     *
     * @param vecSurfaceRef Surface vector in the ref config
     * @param orientation Quaternion specifying the surface's orientation about its core.
     *
     * @return The surface vector in the current config
     */
    EfieldNP3 rotateSurfaceVecRefToCurrentConfig(const EfieldNP3 &vecSurfaceRef, const Equatn &orientation) const;

    /**
     * @brief Map the surface vector at each grid point to the sphere
     *
     * @param vecSurface Surface vector in the current config
     * @param orientation Quaternion specifying the surface's orientation about its core.
     *
     * @return Sphere vector in the ref config
     */
    EfieldNP3 pullbackSurfaceVectors(const EfieldNP3 &vecSurface, const Equatn &orientation) const;

    /**
     * @brief Map the sphere vector at each grid point to the surface
     *
     * @param vecSphere Sphere vector in the ref config
     * @param orientation Quaternion specifying the surface's orientation about its core.
     *
     * @return Surface vector in the current config
     */
    EfieldNP3 pushforwardSphereVectors(const EfieldNP3 &vecSphere, const Equatn &orientation) const;

    /**
     * @brief Compute the spectral decomposition of a surface vector evaluated at each grid point
     *
     * @param vecSurface A surface vector evaluated at each grid point
     *
     * @return vshCoeff The vector spherical harmonic coefficients
     *          (idx 0: radial, idx 1: divFree, idx 2: curlFree)
     */
    EfieldNS3 decomposeSurfaceVectorFcn(const EfieldNP3 &vecSurface, const Equatn &orientation) const;

    /**
     * @brief Compute the reconstruction of a surface vector based on its vector spherical hamonic coefficients
     *
     * @param vshCoeff The vector spherical harmonic coefficients
     *          (idx 0: radial, idx 1: divFree, idx 2: curlFree)
     *
     * @return The reconstructed surface vector evaluated at each grid point
     */
    EfieldNP3 reconstructSurfaceVectorFcn(const EfieldNS3 &vshCoeff const Equatn &orientation) const;

    /**
     * @brief Compute the spectral decomposition of a surface scalar evaluated at each grid point
     *
     * @param scalarSurface A surface scalar function evaluated at each grid point
     *
     * @return The spherical harmonic coefficients of the decomposition
     */
    EfieldNS3 decomposeSurfaceScalarFcn(const EfieldNP3 &scalarSurface) const;

    /**
     * @brief Compute the reconstruction of a surface scalar function based on its spherical hamonic coefficients
     *
     * @param shCoeff The spherical harmonic coefficients of the decomposition
     *
     * @return The reconstructed surface scalar function evaluated at each grid point
     */
    EfieldNP3 reconstructSurfaceScalarFcn(const EfieldNS3 &shCoeff) const;

    /**
     * @brief Compute the L2 norm of a scalar field on the surface
     *
     * @param surfaceScalarField A scalar field on the surface
     *
     * @return The L2 norm of the scalar field on the surface
     */
    double l2normSurfaceScalarField(const EfieldNP &surfaceScalarField) const;

    /**
     * @brief Compute the L2 norm of a vector field on the surface
     *
     * @param surfaceVecField A scalar field on the surface
     *
     * @return The L2 norm of the vector field on the surface
     */
    double l2normSurfaceVecField(const EfieldNP3 &surfaceVecField) const;

    /**
     * @brief Compute the L2 norm of a scalar field on the sphere
     *
     * @param sphereScalarField A scalar field on the sphere
     *
     * @return The L2 norm of the scalar field on the sphere
     */
    double l2normSphereScalarField(const EfieldNP &sphereScalarField) const;

    /**
     * @brief Compute the L2 norm of a vector field on the sphere
     *
     * @param sphereVecField A vector field on the sphere
     *
     * @return The L2 norm of the vector field on the sphere
     */
    double l2normSphereVecField(const EfieldNP3 &sphereVecField) const;

    /**
     * @brief Compute int F G^* = F_i dot G^*_i * w_i
     *
     * @param F Some spherical vector function evaluated at each grid point
     * @param G Some other spherical vector function evaluated at each grid point
     *
     * @return The spherical inner product of F and G
     */
    double sphericalInnerProduct(const EfieldNP3 &F, const EfieldNP3 &G) const;

    /**
     * @brief Compute int F G^* = F_i dot G^*_i * w_i
     *
     * @param F Some spherical scalar function evaluated at each grid point
     * @param G Some other spherical scalar function evaluated at each grid point
     *
     * @return The spherical inner product of F and G
     */
    double sphericalInnerProduct(const EfieldNP &F, const EfieldNP &G) const;

    /**
     * @brief Compute int F G^* = F_i dot G^*_i * w_i
     *
     * @param F Some surface vector function evaluated at each grid point
     * @param G Some other surface vector function evaluated at each grid point
     *
     * @return The surface inner product of F and G
     */
    double surfaceInnerProduct(const EfieldNP3 &F, const EfieldNP3 &G) const;

    /**
     * @brief Compute int F G^* = F_i dot G^*_i * w_i
     *
     * @param F Some surface scalar function evaluated at each grid point
     * @param G Some other surface scalar function evaluated at each grid point
     *
     * @return The surface inner product of F and G
     */
    double surfaceInnerProduct(const EfieldNP &F, const EfieldNP &G) const;

    /**
     * @brief Compute the spectral decomposition of a sphere scalar evaluated at each grid point
     *
     * @param scalarSphere A sphere scalar function evaluated at each grid point
     *
     * @return The spherical harmonic coefficients of the decomposition
     */
    EfieldNScd decomposeSphereScalarFcn(const EfieldNP &scalarSphere) const;

    /**
     * @brief Compute the reconstruction of a sphere scalar function based on its spherical hamonic coefficients
     *
     * @param shCoeff The spherical harmonic coefficients of the decomposition
     *
     * @return The reconstructed sphere scalar function evaluated at each grid point
     */
    EfieldNP reconstructSphereScalarFcn(const EfieldNScd &shCoeff) const;
};

// Include the SharedParticleSurface implimentation
#include "SharedParticleSurface.tpp"

#endif