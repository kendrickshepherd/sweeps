#include <NFormSupportPolicy.hpp>
#include <CombinatorialMapMethods.hpp>
#include <TPCombinatorialMap.hpp>
#include <algorithm>
#include <set>

using namespace basis;

namespace
{
    util::IndexVec cellIndices( const topology::TPCombinatorialMap& cmap, const topology::Cell& c )
    {
        util::IndexVec out;
        for( const topology::Dart& d : topology::unflattenFull( cmap, c.dart() ).unflat_darts ) out.push_back( d.id() );
        return out;
    }

    util::IndexVec cellCounts( const topology::TPCombinatorialMap& cmap )
    {
        util::IndexVec out;
        for( const auto& c : topology::tensorProductComponentCMaps( cmap ) ) out.push_back( topology::cellCount( *c, 1 ) );
        return out;
    }

    void requireTwoDimensional( const topology::CombinatorialMap& cmap )
    {
        if( cmap.dim() != 2 )
            throw NFormSupportNotImplemented( "The n-form support policy is implemented only for 2d spaces." );
    }

    // R_l from the parent relation: the level 0 cells, then the children of each refined cell.
    std::optional<NFormSupportFailure> firstFailure( const HierarchicalTPSplineSpace& ss, const size_t patch )
    {
        const topology::HierarchicalTPCombinatorialMap& cmap = ss.basisComplex().parametricAtlas().cmap();
        const std::vector<std::vector<topology::Cell>> leaves = topology::leafElements( cmap );

        std::vector<topology::Cell> present;
        topology::iterateCellsWhile( *cmap.refinementLevels().at( 0 ), cmap.dim(), [&]( const topology::Cell& c ) {
            present.push_back( c );
            return true;
        } );

        for( size_t level = 0; level + 1 < cmap.numLevels(); level++ )
        {
            const topology::TPCombinatorialMap& level_cmap = *cmap.refinementLevels().at( level );
            const util::IndexVec counts = cellCounts( level_cmap );
            const auto id = [&]( const topology::Cell& c ) { return util::flatten( cellIndices( level_cmap, c ), counts ); };

            std::set<size_t> leaf_ids;
            for( const topology::Cell& c : leaves.at( level ) ) leaf_ids.insert( id( c ) );

            std::set<size_t> refined;
            std::vector<topology::Cell> children;
            for( const topology::Cell& c : present )
            {
                if( leaf_ids.contains( id( c ) ) ) continue;
                refined.insert( id( c ) );
                cmap.iterateChildren( c, level, [&]( const topology::Cell& child ) {
                    children.push_back( child );
                    return true;
                } );
            }

            std::set<size_t> covered;
            for( const auto& support : tensorCellSupports( *ss.refinementLevels().at( level ) ) )
            {
                std::vector<size_t> ids;
                for( const auto& cell : support ) ids.push_back( util::flatten( cell, counts ) );
                if( std::all_of( ids.begin(), ids.end(), [&]( const size_t i ) { return refined.contains( i ); } ) )
                    covered.insert( ids.begin(), ids.end() );
            }

            for( const size_t c : refined )
                if( not covered.contains( c ) ) return NFormSupportFailure{ patch, level, util::unflatten( c, counts ) };

            present = std::move( children );
        }
        return std::nullopt;
    }
}

namespace basis
{
    std::optional<NFormSupportFailure> checkNFormSupportPolicy( const HierarchicalTPSplineSpace& ss )
    {
        requireTwoDimensional( ss.basisComplex().parametricAtlas().cmap() );
        return firstFailure( ss, 0 );
    }

    std::optional<NFormSupportFailure> checkNFormSupportPolicy( const HierarchicalMultiPatchSplineSpace& ss )
    {
        requireTwoDimensional( ss.basisComplex().parametricAtlas().cmap() );
        for( size_t patch = 0; patch < ss.constituents().size(); patch++ )
            if( auto failure = firstFailure( *ss.constituents().at( patch ), patch ) ) return failure;
        return std::nullopt;
    }
}
