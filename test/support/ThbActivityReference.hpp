#pragma once
#include <Cell.hpp>
#include <FunctionId.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace thb_test
{
    // Frozen from basis::detail::levelActiveFunctions at b99440c (before optimization C).
    // Deliberately independent of ThbActiveFunctions.hpp. Do not update this when optimizing production.
    // This is an equivalence reference, not an independent proof of THB support containment.
    template <typename HierarchicalCMap, typename LevelSpace>
    std::set<basis::FunctionId> referenceLevelActiveFunctions(
        const HierarchicalCMap& cmap,
        const std::vector<std::vector<topology::Cell>>& leaf_elements,
        const size_t level,
        const LevelSpace& level_ss )
    {
        std::set<basis::FunctionId> active;
        for( const topology::Cell& leaf : leaf_elements.at( level ) )
        {
            const std::vector<basis::FunctionId> conn = level_ss.connectivity( leaf );
            active.insert( conn.begin(), conn.end() );
        }

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
            const std::vector<basis::FunctionId> conn = level_ss.connectivity( c );
            for( const basis::FunctionId& fid : conn ) active.erase( fid );
        }
        return active;
    }

    inline void requireSameActiveFunctions( const std::set<basis::FunctionId>& expected,
                                            const std::set<basis::FunctionId>& actual,
                                            const std::string& context )
    {
        if( expected == actual ) return;
        std::ostringstream message;
        message << context << ": active function mismatch; expected " << expected.size() << ", got " << actual.size();
        size_t shown = 0;
        for( const auto& fid : expected )
            if( not actual.contains( fid ) and shown++ < 8 ) message << "; missing " << fid.id();
        shown = 0;
        for( const auto& fid : actual )
            if( not expected.contains( fid ) and shown++ < 8 ) message << "; extra " << fid.id();
        throw std::runtime_error( message.str() );
    }
}
