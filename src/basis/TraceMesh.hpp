#pragma once
#include <TraceExtraction.hpp>
#include <MultiPatchCombinatorialMap.hpp>
#include <SideCoordinateTransform.hpp>
#include <Eigen/Core>
#include <optional>
#include <vector>

namespace basis
{
    enum class TraceMeshEntityType
    {
        Boundary,
        PatchInterface,
        Interior
    };

    enum class InteriorTraceMode
    {
        All,
        Broken
    };

    /// Places a common interface fragment in a leaf side.  With xi the fragment
    /// coordinate and y the leaf side coordinate, both on the unit cube in
    /// tangentialAxes order, and s = axes.source_axis_for_destination[d]:
    ///   y_d = offset_d + scale_d * ( axes.source_axis_reversed[d] ? 1 - xi_s : xi_s )
    struct TraceSideEmbedding
    {
        topology::SideCoordinateTransform axes;
        SmallVector<double, 2> offset;
        SmallVector<double, 2> scale;
    };

    /// The fragment is the whole leaf side, in the leaf side's own axes.
    TraceSideEmbedding wholeTraceSideEmbedding( size_t rank );

    /// Throws unless the axes are a signed permutation, offset and scale share
    /// their rank and are finite, every scale is positive, and the fragment lies
    /// in the leaf side to within tol.
    void validateTraceSideEmbedding( const TraceSideEmbedding& embedding, double tol = 1e-10 );

    Eigen::VectorXd embedTraceFragmentPoint( const TraceSideEmbedding& embedding, const Eigen::VectorXd& xi );

    struct TraceMeshSide
    {
        /// In a hierarchical space, one of the leaf's darts, unspecified which.
        /// It may belong to a finer refinement level than `level`, which
        /// identifies the element's unrefined ancestor.
        topology::Cell element;
        ElementSide side;
        size_t patch_id = 0;
        size_t level = 0;
        TraceSideEmbedding embedding;
    };

    /// One common interface fragment.  Each side names the leaf whose side
    /// contains the fragment and embeds the fragment in that leaf side.  The
    /// fragment coordinate follows the first side's tangential axes, so
    /// first.embedding.axes is the identity and, on a patch interface,
    /// second->embedding.axes is the interface's sideCoordinateTransform.
    struct TraceMeshInterface
    {
        TraceMeshEntityType type;
        TraceMeshSide first;
        std::optional<TraceMeshSide> second;
        std::optional<topology::MultiPatchCombinatorialMap::TPPermutation> permutation;

        bool isBoundary() const { return not second.has_value(); }
    };

    /// Throws unless both embeddings are valid with one rank, the record has
    /// its type's shape (boundary: no second side; interior: a second side;
    /// patch interface: a second side and a permutation; only a patch
    /// interface has a permutation), boundary and interior records embed whole
    /// sides by the identity, and a patch interface record uses the fragment
    /// frame above.
    void validateTraceMeshInterface( const TraceMeshInterface& iface, double tol = 1e-10 );

    std::vector<size_t> permuteTraceSideIndex(
        const topology::SideCoordinateTransform& transform,
        const std::vector<size_t>& index,
        const std::vector<size_t>& lengths );

    Eigen::VectorXd permuteTraceSidePoint(
        const topology::SideCoordinateTransform& transform,
        const Eigen::VectorXd& point );

    std::vector<TraceMeshInterface> boundaryTraceMeshInterfaces( const SplineSpace& ss );

    std::vector<TraceMeshInterface> patchTraceMeshInterfaces( const SplineSpace& ss );

    /// Enumerates same-patch, same-level leaf interfaces, each once, with the first
    /// side on the max side of the shared face.  In hierarchical spaces this
    /// intentionally skips coarse-fine subdivision; that belongs with the physical trace
    /// integration layer, where quadrature-side subdivision can be handled explicitly.
    std::vector<TraceMeshInterface> interiorTraceMeshInterfaces(
        const SplineSpace& ss,
        InteriorTraceMode mode = InteriorTraceMode::All,
        double row_tol = 1e-12 );

    std::vector<TraceMeshInterface> traceMeshInterfaces(
        const SplineSpace& ss,
        bool include_boundary,
        bool include_patch_interfaces,
        bool include_interior,
        InteriorTraceMode interior_mode = InteriorTraceMode::All,
        double row_tol = 1e-12 );

    /// Extracts each named leaf's whole side.  The embeddings are not applied:
    /// evaluation or quadrature on the fragment must map its point through
    /// iface.first.embedding and iface.second->embedding.
    TraceInterfaceElement traceInterfaceElement(
        const SplineSpace& ss,
        const TraceMeshInterface& iface,
        double row_tol = 1e-12 );
}
