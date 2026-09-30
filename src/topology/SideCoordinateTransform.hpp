#pragma once
#include <MultiPatchCombinatorialMap.hpp>
#include <SmallVector.hpp>
#include <vector>

namespace topology
{
    // The correspondence between the tangential coordinates of two glued
    // tensor-product patch sides: F_b o P o F_a reduced to a signed
    // permutation, where F_s is the axis-0 reversal that dropToBoundary and
    // raiseFromBoundary each apply once and P is the stored TPPermutation.
    // Both arrays are indexed by DESTINATION tangential axis.
    struct SideCoordinateTransform
    {
        SmallVector<size_t, 2> source_axis_for_destination;
        SmallVector<bool, 2> source_axis_reversed;
    };

    // For destination tangential axis d, with s = source_axis_for_destination[d]
    // and n_s the SOURCE extent of axis s:
    //   j_d = source_axis_reversed[d] ? n_s - 1 - i_s : i_s
    SideCoordinateTransform sideCoordinateTransform( size_t dim,
                                                     size_t source_side_id,
                                                     size_t destination_side_id,
                                                     MultiPatchCombinatorialMap::TPPermutation permutation );

    // The destination side shape the transform implies.  Compare this against
    // the actual destination extents to check interface conformity.
    std::vector<size_t> transformedExtents( const SideCoordinateTransform& transform,
                                            const std::vector<size_t>& source_extents );

    std::vector<size_t> transformSideIndex( const SideCoordinateTransform& transform,
                                            const std::vector<size_t>& source_index,
                                            const std::vector<size_t>& source_extents );

    // Side coordinates on the unit square; a reversed axis becomes 1 - x.
    std::vector<double> transformSidePoint( const SideCoordinateTransform& transform,
                                            const std::vector<double>& source_point );
}
