#ifndef PARTICLESHAPES_HPP_
#define PARTICLESHAPES_HPP_

#include "Util/YamlHelper.hpp"

/**
 * @brief Class for calculating information specific to 3D genus zero shapes and their surfaces
 * Because each shape is genus zero, we can utalize its homeomorphism to the unit sphere
 * All calculations assume that this homeomorphism is known exactly
 *
 */
class ParticleShapes {
  private:
    /**
     * @brief Compute the normalized vector from the sphere center to the sphere surface (identical to the sphere normal)
     *
     * @param thetaA azimuthal angle with z-axis pole
     * @param phiA polar angle with z-axis pole
     *  
     * @return Sphere normal
     */
    Evec3 calcSurfacePointA(const double thetaA, const double phiA) const;

    /**
     * @brief Compute the normalized vector from the sphere center to the sphere surface (identical to the sphere normal)
     *
     * @param thetaB azimuthal angle with x-axis pole
     * @param phiB polar angle with x-axis pole
     *  
     * @return Sphere normal
     */
    Evec3 calcSurfacePointB(const double thetaB, const double phiB) const;

    /**
     * @brief Compute the partial derivative of the sphere normal with respect to thetaA
     *
     * @param thetaA azimuthal angle with z-axis pole
     * @param phiA polar angle with z-axis pole
     *  
     * @return Partial sphere normal partial thetaA
     */
    Evec3 calcPSphereVectorHatPThetaA(const double thetaA, const double phiA) const;

    /**
     * @brief Compute the partial derivative of the sphere normal with respect to thetaB
     *
     * @param thetaB azimuthal angle with x-axis pole
     * @param phiB polar angle with x-axis pole
     *  
     * @return Partial sphere normal partial thetaB
     */
    Evec3 calcPSphereVectorHatPThetaB(const double thetaB, const double phiB) const;

    /**
     * @brief Compute the partial derivative of the sphere normal with respect to phiA
     *
     * @param thetaA azimuthal angle with z-axis pole
     * @param phiA polar angle with z-axis pole
     *  
     * @return Partial sphere normal partial phiA
     */
    Evec3 calcPSphereVectorHatPPhiA(const double thetaA, const double phiA) const;

    /**
     * @brief Compute the partial derivative of the sphere normal with respect to phiB
     *
     * @param thetaB azimuthal angle with x-axis pole
     * @param phiB polar angle with x-axis pole
     *  
     * @return Partial sphere normal partial phiB
     */
    Evec3 calcPSphereVectorHatPPhiB(const double thetaB, const double phiB) const;

    /**
     * @brief Calculate the sphere basis  
     *   this includes the sphere normal, partial sphere normal partial thetaA, 
     *   and partial sphere normal partial phiA
     *
     * @param thetaA azimuthal angle with z-axis pole
     * @param phiA polar angle with z-axis pole
     *  
     * @return [sphere normal, partial sphere normal partial thetaA, partial sphere normal partial phiA]
     */
    Evec3 calcSphereBasisVectorsA(const double thetaA, const double phiA) const;

    /**
     * @brief Calculate the sphere basis  
     *  this includes the sphere normal, partial sphere normal partial thetaA, 
     *  and partial sphere normal partial phiA
     *
     * @param thetaB azimuthal angle with x-axis pole
     * @param phiB polar angle with x-axis pole
     *  
     * @return [sphere normal, partial sphere normal partial thetaB, partial sphere normal partial phiB]
     */
    Evec3 calcSphereBasisVectorsB(const double thetaB, const double phiB) const;


  public:
    /**
     * @brief Construct a new ParticleShapes object
     *
     */
    ParticleShapes() = default;

    /**
     * @brief Print the shape properties
     */
    virtual void echo() = 0;

    /**
     * @brief Initialize from a yaml node
     *
     * @param config
     */
    virtual void initialize(const YAML::Node &config) = 0;

    /**
     * @brief Get the unique ID of the shape
     *
     * @return shapeID
     */
    virtual int getShapeID() = 0;

    /**
     * @brief Calculate the radius at the given thetaA and phiA
     *
     * @param thetaA Azimuthal angle with z-axis pole
     * @param phiA Polar angle with z-axis pole
     * 
     * @return radius
     */
    virtual double calcRadiusA(const double thetaA, const double phiA) = 0;

    /**
     * @brief Calculate the radius at the given thetaB and phiB
     *
     * @param thetaB azimuthal angle with x-axis pole
     * @param phiB polar angle with x-axis pole
     * 
     * @return radius
     */
    virtual double calcRadiusB(const double thetaB, const double phiB) = 0;

    /**
     * @brief Apply the mapping from a point in the sphere to a unique point in the surface
     *
     * @param spherePoint A point in or on the sphere
     * 
     * @return surfacePoint
     */
    virtual Evec3 pushforwardPoint(const Evec3 spherePoint) = 0;

    /**
     * @brief Compute the pushforward matrix P that maps a vector in the sphere to a vector in the surface
     *
     * @param point A point on the sphere
     * 
     * @return flattened 3x3 pushforward matrix
     */
    virtual Emat3 calcPushforwardMatrix(const Evec3 point) const = 0;

    /**
     * @brief Compute the pullback matrix P^{-1} that maps a vector in the surface to a vector in the sphere
     *
     * @param point A point on the sphere
     * 
     * @return flattened 3x3 pullback matrix
     */
    virtual Emat3 calcPullbackMatrix(const Evec3 point) const = 0;

    /**
     * @brief Compute the 3x3 pushforward matrix $P$ that maps a vector in the sphere to a vector in the surface
     *
     * @param thetaA azimuthal angle with z-axis pole
     * @param phiA polar angle with z-axis pole
     * @param thetaB azimuthal angle with x-axis pole
     * @param phiB polar angle with x-axis pole
     * 
     * @return flattened 3x3 pushforward matrix
     */
    Evec3 calcPushforwardMatrix(const double thetaA, const double phiA, const double thetaB, const double phiB); 

    /**
     * @brief Compute the 3x3 pullback matrix $P^{-1}$ that maps a vector in the surface to a vector in the sphere
     *
     * @param thetaA azimuthal angle with z-axis pole
     * @param phiA polar angle with z-axis pole
     * @param thetaB azimuthal angle with x-axis pole
     * @param phiB polar angle with x-axis pole
     * 
     * @return flattened 3x3 pushforward matrix
     */
    Emat3 calcPullbackMatrix(const double thetaA, const double phiA, const double thetaB, const double phiB);

    /**
     * @brief Calculate the surface point corresponding to (thetaA, phiA) on the sphere 
     *
     * @param thetaA azimuthal angle with z-axis pole
     * @param phiA polar angle with z-axis pole
     * 
     * @return The surface point corresponding to (thetaA, phiA)
     */
    Evec3 calcSurfacePointA(const double thetaA, const double phiA) const;

    /**
     * @brief Calculate the surface point corresponding to (thetaB, phiB) on the sphere 
     *
     * @param thetaB azimuthal angle with x-axis pole
     * @param phiB polar angle with x-axis pole
     * 
     * @return The surface point corresponding to (thetaB, phiB)
     */
    Evec3 calcSurfacePointB(const double thetaB, const double phiB) const;

    /**
     * @brief Compute the normalized surface normal by the cross product of the pushed-forward sphere basis vectors. 
     *  The surface weight is obtained using the magnatude of the sphere normal (before normalization). 
     *
     * @param thetaA azimuthal angle with z-axis pole
     * @param phiA polar angle with z-axis pole
     * @param thetaB azimuthal angle with x-axis pole
     * @param phiB polar angle with x-axis pole
     * @param sphereWeight quadrature weight at the given point on the sphere
     *  
     * @return [normalized surface normal, surface weight]
     */
    void calcSurfaceNormalHatAndWeight(const double thetaA, const double phiA, const double thetaB, const double phiB, const double sphereWeight) const;

    /**
     * @brief Compute thetaA, phiA from a point on the sphere
     *
     * @param point A point on the sphere
     *  
     * @return [thetaA, phiA]
     */
    void calcThetaPhiAFromPoint(const Evec3 point) const;

    /**
     * @brief Compute thetaB, phiB from a point on the sphere
     *
     * @param point A point on the sphere
     *  
     * @return [thetaB, phiB]
     */
    void calcThetaPhiBFromPoint(const Evec3 point) const;

};

class Sphere : public ParticleShapes {
  public:
    Sphere(const YAML::Node &config) { initialize(config); };
    Sphere(const double radius_);
    virtual ~Sphere() = default;
    virtual void initialize(const YAML::Node &config);
    virtual int getShapeID() {return 0};
    virtual double calcRadiusA(const double thetaA, const double phiA) const;
    virtual double calcRadiusB(const double thetaB, const double phiB) const;
    virtual Evec3 pushforwardPoint(const Evec3 spherePoint) const;
    virtual Emat3 calcPushforwardMatrix(const Evec3 spherePoint) const;
    virtual Emat3 calcPullbackMatrix(const Evec3 spherePoint) const;
    virtual void echo() const;

  private:
    double radius = 1.0;
};

class Spheroid : public ParticleShapes {
  public:
    Spheroid(const YAML::Node &config) { initialize(config); };
    Spheroid(const double axisLenX_, const double axisLenY_, const double axisLenZ_]);
    ~Spheroid() = default;
    void initialize(const YAML::Node &config);
    virtual int getShapeID() {return 1};
    virtual double calcRadiusA(const double thetaA, const double phiA) const;
    virtual double calcRadiusB(const double thetaB, const double phiB) const;
    virtual Evec3 pushforwardPoint(const Evec3 spherePoint) const;
    virtual Emat3 calcPushforwardMatrix(const Evec3 spherePoint) const;
    virtual Emat3 calcPullbackMatrix(const Evec3 spherePoint) const;
    virtual void echo() const;

  private:
    double axisLenX = 1.0;
    double axisLenY = 2.0;
    double axisLenZ = 3.0;
};

class Helix : public ParticleShapes {
  public:
    Helix(const YAML::Node &config) { initialize(config); };
    Helix(const double height_, const double majorRadius_, const double minorRadius_, const double frequency_);
    ~Helix() = default;
    void initialize(const YAML::Node &config);
    virtual int getShapeID() {return 2};
    virtual double calcRadiusA(const double thetaA, const double phiA) const;
    virtual double calcRadiusB(const double thetaB, const double phiB) const;
    virtual Evec3 pushforwardPoint(const Evec3 spherePoint) const;
    virtual Emat3 calcPushforwardMatrix(const Evec3 spherePoint) const;
    virtual Emat3 calcPullbackMatrix(const Evec3 spherePoint) const;
    virtual void echo() const;

  private:
    double height = 1.0;
    double majorRadius = 0.5;
    double minorRadius = 0.5;
    double frequency = 4.0;
};

#endif