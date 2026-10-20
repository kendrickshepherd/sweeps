#pragma once
#include <set>
#include <utility>
#include <vector>
#include <Cell.hpp>
#include <FunctionId.hpp>

// Internal to the hierarchical spline spaces; not included by any public header.
namespace basis::detail
{
    /// @brief The THB active functions of one refinement level: supp in Omega^level and not in
    /// Omega^(level+1).
    /// @param cmap           A hierarchical cmap providing iterateChildren( level cell, level, callback ).
    /// @param leaf_elements  Level-local leaf elements, per level.
    /// @param level_ss       The level space, providing connectivity( level cell ).
    template <typename HierarchicalCMap, typename LevelSpace>
    std::set<FunctionId> levelActiveFunctions( const HierarchicalCMap& cmap,
                                               const std::vector<std::vector<topology::Cell>>& leaf_elements,
                                               const size_t level,
                                               const LevelSpace& level_ss )
    {
        std::set<FunctionId> active;
        for( const topology::Cell& leaf : leaf_elements.at( level ) )
        {
            const std::vector<FunctionId> conn = level_ss.connectivity( leaf );
            active.insert( conn.begin(), conn.end() );
        }

        // Level cells covered by a coarser leaf, i.e. outside Omega^level.
        std::vector<topology::Cell> covered;
        for( size_t coarse = 0; coarse < level; coarse++ )
        {
            covered.insert( covered.end(), leaf_elements.at( coarse ).begin(), leaf_elements.at( coarse ).end() );
            std::vector<topology::Cell> children;
            for( const topology::Cell& c : covered )
                cmap.iterateChildren( c, coarse, [&]( const topology::Cell& child ) {
                    children.push_back( child );
                    return true;
                } );
            covered = std::move( children );
        }

        for( const topology::Cell& c : covered )
        {
            const std::vector<FunctionId> conn = level_ss.connectivity( c );
            for( const FunctionId& fid : conn ) active.erase( fid );
        }
        return active;
    }
}
