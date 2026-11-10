#pragma once
#include <VectorConformingTPSplineSpace.hpp>
#include <SplineSpaceEvaluator.hpp>
#include <CombinatorialMapBoundary.hpp>
#include <VectorConformingHierarchicalTPSplineSpace.hpp>
#include <VectorConformingHierarchicalMultiPatchSplineSpace.hpp>
#include <IndexOperations.hpp>

namespace api
{
    class NavierStokesDiscretization
    {
        public:

        virtual ~NavierStokesDiscretization() = default;

        virtual const Eigen::MatrixXd& geometryControlPoints() const = 0;
        virtual const Eigen::MatrixXd& controlPoints() const { return geometryControlPoints(); }
        virtual bool hasRationalGeometry() const { return false; }

        virtual const topology::CombinatorialMapBoundary& cmapBdry() const = 0;

        virtual eval::SplineSpaceEvaluator& getH1() = 0;
        virtual eval::SplineSpaceEvaluator& getHDIV() = 0;
        virtual eval::SplineSpaceEvaluator& getL2() = 0;
        virtual eval::SplineSpaceEvaluator& getGeometry() { return getH1(); }

        virtual const eval::SplineSpaceEvaluator& getH1() const = 0;
        virtual const eval::SplineSpaceEvaluator& getHDIV() const = 0;
        virtual const eval::SplineSpaceEvaluator& getL2() const = 0;
        virtual const eval::SplineSpaceEvaluator& getGeometry() const { return getH1(); }
    };

    class NavierStokesTPDiscretization : public NavierStokesDiscretization
    {
        public:
        NavierStokesTPDiscretization( const basis::KnotVector& kv_s,
                                    const basis::KnotVector& kv_t,
                                    const size_t degree_s,
                                    const size_t degree_t,
                                    const Eigen::MatrixXd& cpts );

        virtual ~NavierStokesTPDiscretization() = default;

        virtual const Eigen::MatrixXd& geometryControlPoints() const override { return cpts; }

        virtual const topology::CombinatorialMapBoundary& cmapBdry() const override { return cmap_bdry; }

        virtual eval::SplineSpaceEvaluator& getH1() override { return H1; }
        virtual eval::SplineSpaceEvaluator& getHDIV() override { return HDIV; }
        virtual eval::SplineSpaceEvaluator& getL2() override { return L2; }

        virtual const eval::SplineSpaceEvaluator& getH1() const override { return H1; }
        virtual const eval::SplineSpaceEvaluator& getHDIV() const override { return HDIV; }
        virtual const eval::SplineSpaceEvaluator& getL2() const override { return L2; }

        const basis::TPSplineSpace H1_ss;
        const basis::VectorConformingTPSplineSpace HDIV_ss;
        const basis::TPSplineSpace L2_ss;

        private:

        const topology::CombinatorialMapBoundary cmap_bdry;

        const Eigen::MatrixXd cpts;

        eval::SplineSpaceEvaluator H1;
        eval::SplineSpaceEvaluator HDIV;
        eval::SplineSpaceEvaluator L2;
    };

    class NURBSNavierStokesTPDiscretization : public NavierStokesTPDiscretization
    {
        public:
        NURBSNavierStokesTPDiscretization( const basis::KnotVector& kv_s,
                                           const basis::KnotVector& kv_t,
                                           const size_t degree_s,
                                           const size_t degree_t,
                                           const Eigen::MatrixXd& control_points,
                                           const Eigen::VectorXd& weights );

        NURBSNavierStokesTPDiscretization( const basis::KnotVector& kv_s,
                                           const basis::KnotVector& kv_t,
                                           const size_t degree_s,
                                           const size_t degree_t,
                                           const Eigen::MatrixXd& homogeneous_cpts );

        virtual ~NURBSNavierStokesTPDiscretization() = default;

        virtual const Eigen::MatrixXd& controlPoints() const override { return euclidean_cpts; }
        virtual bool hasRationalGeometry() const override { return true; }

        virtual eval::SplineSpaceEvaluator& getGeometry() override { return Geometry; }
        virtual const eval::SplineSpaceEvaluator& getGeometry() const override { return Geometry; }

        const Eigen::VectorXd& weights() const { return cpt_weights; }

        private:
        const Eigen::MatrixXd euclidean_cpts;
        const Eigen::VectorXd cpt_weights;
        eval::NURBSSpaceEvaluator Geometry;
    };

    enum class PatchSide
    {
        S0,
        S1,
        T0,
        T1
    };

    class NavierStokesHierarchicalDiscretization : public NavierStokesDiscretization
    {
        public:
        NavierStokesHierarchicalDiscretization( const basis::KnotVector& kv_s,
                                                const basis::KnotVector& kv_t,
                                                const size_t degree_s,
                                                const size_t degree_t,
                                                const Eigen::MatrixXd& unrefined_cpts,
                                                const std::vector<std::vector<std::pair<size_t, size_t>>>& elems_to_refine );

        virtual ~NavierStokesHierarchicalDiscretization() = default;

        virtual const Eigen::MatrixXd& geometryControlPoints() const override { return cpts; }

        virtual const topology::CombinatorialMapBoundary& cmapBdry() const override { return cmap_bdry; }

        virtual eval::SplineSpaceEvaluator& getH1() override { return H1; }
        virtual eval::SplineSpaceEvaluator& getHDIV() override { return HDIV; }
        virtual eval::SplineSpaceEvaluator& getL2() override { return L2; }

        virtual const eval::SplineSpaceEvaluator& getH1() const override { return H1; }
        virtual const eval::SplineSpaceEvaluator& getHDIV() const override { return HDIV; }
        virtual const eval::SplineSpaceEvaluator& getL2() const override { return L2; }

        const basis::HierarchicalTPSplineSpace H1_ss;
        const basis::VectorConformingHierarchicalTPSplineSpace HDIV_ss;
        const basis::HierarchicalTPSplineSpace L2_ss;

        private:

        const topology::CombinatorialMapBoundary cmap_bdry;

        const Eigen::MatrixXd cpts;

        eval::SplineSpaceEvaluator H1;
        eval::SplineSpaceEvaluator HDIV;
        eval::SplineSpaceEvaluator L2;
    };

    class NURBSNavierStokesHierarchicalDiscretization : public NavierStokesHierarchicalDiscretization
    {
        public:
        NURBSNavierStokesHierarchicalDiscretization(
            const basis::KnotVector& kv_s,
            const basis::KnotVector& kv_t,
            const size_t degree_s,
            const size_t degree_t,
            const Eigen::MatrixXd& control_points,
            const Eigen::VectorXd& weights,
            const std::vector<std::vector<std::pair<size_t, size_t>>>& elems_to_refine );

        NURBSNavierStokesHierarchicalDiscretization(
            const basis::KnotVector& kv_s,
            const basis::KnotVector& kv_t,
            const size_t degree_s,
            const size_t degree_t,
            const Eigen::MatrixXd& homogeneous_cpts,
            const std::vector<std::vector<std::pair<size_t, size_t>>>& elems_to_refine );

        virtual ~NURBSNavierStokesHierarchicalDiscretization() = default;

        virtual const Eigen::MatrixXd& controlPoints() const override { return euclidean_cpts; }
        virtual bool hasRationalGeometry() const override { return true; }

        virtual eval::SplineSpaceEvaluator& getGeometry() override { return Geometry; }
        virtual const eval::SplineSpaceEvaluator& getGeometry() const override { return Geometry; }

        const Eigen::VectorXd& weights() const { return cpt_weights; }

        private:
        const Eigen::MatrixXd euclidean_cpts;
        const Eigen::VectorXd cpt_weights;
        eval::NURBSSpaceEvaluator Geometry;
    };

    class AdaptiveLocalNavierStokesHierarchicalDiscretization : public NavierStokesHierarchicalDiscretization
    {
        public:
        using NavierStokesHierarchicalDiscretization::NavierStokesHierarchicalDiscretization;
        virtual ~AdaptiveLocalNavierStokesHierarchicalDiscretization() = default;
    };

    class NURBSAdaptiveLocalNavierStokesHierarchicalDiscretization : public NURBSNavierStokesHierarchicalDiscretization
    {
        public:
        using NURBSNavierStokesHierarchicalDiscretization::NURBSNavierStokesHierarchicalDiscretization;
        virtual ~NURBSAdaptiveLocalNavierStokesHierarchicalDiscretization() = default;
    };

    struct PatchGeometry
    {
        basis::KnotVector kv_s;
        basis::KnotVector kv_t;
        size_t degree_s;
        size_t degree_t;
        /// ( spatial dim, functions of the unrefined patch space ).
        Eigen::MatrixXd control_points;
    };

    struct PatchSideRef
    {
        size_t patch;
        PatchSide side;
    };

    /// Patch-local tensor element indices ( i, j ) on one level.
    struct PatchElement
    {
        size_t patch;
        size_t i;
        size_t j;
    };

    /// @brief A 2d hierarchical multipatch discretization.  Level l + 1 is the dyadic refinement of level l on
    /// every patch; the strong interface checks run on every level.
    class NavierStokesHierarchicalMultiPatchDiscretization : public NavierStokesDiscretization
    {
        public:
        /// @param gluing  Glued side pairs, each side at most once; in 2d the sides fix the orientation.
        /// @param elems_to_refine  elems_to_refine[l]: the level l leaves to refine.
        /// @throws std::out_of_range if a patch or element index is outside the discretization.
        /// @throws std::runtime_error if an element to refine is not a leaf on its level.
        /// @throws std::invalid_argument for malformed gluing or geometry, or patch geometries that disagree on a
        ///         shared H1 coefficient (shared_coefficient_tolerance).
        NavierStokesHierarchicalMultiPatchDiscretization(
            const std::vector<PatchGeometry>& patches,
            const std::vector<std::pair<PatchSideRef, PatchSideRef>>& gluing,
            const std::vector<std::vector<PatchElement>>& elems_to_refine );

        virtual ~NavierStokesHierarchicalMultiPatchDiscretization() = default;

        /// Two patches' coefficients a, b of a shared H1 function agree iff
        /// |a - b| <= shared_coefficient_tolerance * max( 1, |a|, |b| ): absolute below magnitude 1, relative above.
        static constexpr double shared_coefficient_tolerance = 1e-10;

        virtual const Eigen::MatrixXd& geometryControlPoints() const override { return cpts; }

        virtual const topology::CombinatorialMapBoundary& cmapBdry() const override { return cmap_bdry; }

        virtual eval::SplineSpaceEvaluator& getH1() override { return H1; }
        virtual eval::SplineSpaceEvaluator& getHDIV() override { return HDIV; }
        virtual eval::SplineSpaceEvaluator& getL2() override { return L2; }

        virtual const eval::SplineSpaceEvaluator& getH1() const override { return H1; }
        virtual const eval::SplineSpaceEvaluator& getHDIV() const override { return HDIV; }
        virtual const eval::SplineSpaceEvaluator& getL2() const override { return L2; }

        const basis::HierarchicalMultiPatchSplineSpace H1_ss;
        const basis::VectorConformingHierarchicalMultiPatchSplineSpace HDIV_ss;
        const basis::HierarchicalMultiPatchSplineSpace L2_ss;

        private:

        const topology::CombinatorialMapBoundary cmap_bdry;

        const Eigen::MatrixXd cpts;

        eval::SplineSpaceEvaluator H1;
        eval::SplineSpaceEvaluator HDIV;
        eval::SplineSpaceEvaluator L2;
    };

}
