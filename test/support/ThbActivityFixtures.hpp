#pragma once
#include <MultiPatchSplineFactory.hpp>
#include <SideCoordinateTransform.hpp>
#include <CombinatorialMapMethods.hpp>
#include <functional>
#include <random>
#include <set>
#include <string>

namespace thb_test
{
    using Index = std::vector<size_t>;
    using Permutation = topology::MultiPatchCombinatorialMap::TPPermutation;
    using LeafIndices = std::vector<std::vector<std::set<Index>>>; // [level][patch]

    struct FixtureSpec
    {
        size_t dim = 2;
        size_t degree = 2;
        size_t patches = 1;
        Index cells{ 2, 2 };
        std::vector<Index> ratios{ { 2, 2 }, { 2, 2 } };
        std::string marks = "random";
        uint32_t seed = 1729;
        bool reverse = false;
        bool c0 = false;
        Permutation permutation = Permutation::Flip1d;
    };

    inline void eachIndex( const Index& counts, const std::function<void( const Index& )>& call )
    {
        Index index( counts.size(), 0 );
        while( true )
        {
            call( index );
            size_t axis = 0;
            while( axis < counts.size() and ++index.at( axis ) == counts.at( axis ) ) index.at( axis++ ) = 0;
            if( axis == counts.size() ) break;
        }
    }

    inline basis::KnotVector fixtureKnots( const size_t cells, const size_t degree, const bool c0 )
    {
        std::vector<double> knots( degree + 1, 0.0 );
        for( size_t i = 1; i < cells; i++ )
            knots.insert( knots.end(), c0 ? degree : 1, static_cast<double>( i * i ) );
        knots.insert( knots.end(), degree + 1, static_cast<double>( cells * cells ) );
        return basis::KnotVector( knots, 1e-12 );
    }

    inline basis::KnotVector reflectKnots( const basis::KnotVector& kv )
    {
        std::vector<double> knots;
        const double ends = kv.knot( 0 ) + kv.knot( kv.size() - 1 );
        for( size_t i = 0; i < kv.size(); i++ ) knots.push_back( ends - kv.knot( kv.size() - 1 - i ) );
        return basis::KnotVector( knots, 1e-12 );
    }

    struct FixtureLevels
    {
        std::vector<std::vector<std::shared_ptr<const basis::TPSplineSpace>>> patches; // [level][patch]
        std::vector<std::vector<Index>> ratios; // [transition][patch][axis]
        LeafIndices leaves;
        topology::MultiPatchCombinatorialMap::InternalConnectionsMap connections;
    };

    inline FixtureLevels fixtureLevels( const FixtureSpec& spec )
    {
        FixtureLevels out;
        out.ratios.assign( spec.ratios.size(), std::vector<Index>( spec.patches ) );
        for( size_t l = 0; l < spec.ratios.size(); l++ )
            for( auto& ratio : out.ratios.at( l ) ) ratio = spec.ratios.at( l );
        std::vector<SmallVector<basis::KnotVector, 3>> knots( spec.patches );
        SmallVector<size_t, 3> degrees;
        for( size_t axis = 0; axis < spec.dim; axis++ )
        {
            degrees.push_back( spec.degree );
            for( auto& patch : knots ) patch.push_back( fixtureKnots( spec.cells.at( axis ), spec.degree, spec.c0 ) );
        }
        using Side = topology::MultiPatchCombinatorialMap::ConstituentSide;
        const auto connect = [&]( const Side& a, const Side& b, const Permutation p ) {
            out.connections.emplace( a, std::make_pair( p, b ) );
            out.connections.emplace( b, std::make_pair( p, a ) );
        };
        if( spec.patches == 2 )
        {
            const size_t dest_side = spec.reverse ? 0 : 1;
            const auto transform = topology::sideCoordinateTransform( spec.dim, 0, dest_side, spec.permutation );
            for( size_t d = 0; d < spec.dim - 1; d++ )
            {
                const size_t source = transform.source_axis_for_destination.at( d ) + 1;
                knots.at( 1 ).at( d + 1 ) = transform.source_axis_reversed.at( d )
                    ? reflectKnots( knots.at( 0 ).at( source ) ) : knots.at( 0 ).at( source );
                for( size_t l = 0; l < spec.ratios.size(); l++ )
                    out.ratios.at( l ).at( 1 ).at( d + 1 ) = spec.ratios.at( l ).at( source );
            }
            connect( { 0, 0 }, { 1, dest_side }, spec.permutation );
        }
        else if( spec.patches == 4 )
        {
            connect( { 0, 0 }, { 1, 1 }, spec.permutation );
            connect( { 0, 2 }, { 2, 3 }, spec.permutation );
            connect( { 1, 2 }, { 3, 3 }, spec.permutation );
            connect( { 2, 0 }, { 3, 1 }, spec.permutation );
        }

        for( size_t level = 0; level <= spec.ratios.size(); level++ )
        {
            out.patches.emplace_back();
            for( size_t patch = 0; patch < spec.patches; patch++ )
            {
                out.patches.back().push_back( std::make_shared<const basis::TPSplineSpace>(
                    basis::buildBSpline( knots.at( patch ), degrees ) ) );
                if( level == spec.ratios.size() ) continue;
                for( size_t axis = 0; axis < spec.dim; axis++ )
                    knots.at( patch ).at( axis ) = basis::nAdicRefine(
                        knots.at( patch ).at( axis ), out.ratios.at( level ).at( patch ).at( axis ) );
            }
        }

        out.leaves.assign( out.patches.size(), std::vector<std::set<Index>>( spec.patches ) );
        for( size_t patch = 0; patch < spec.patches; patch++ )
        {
            Index counts;
            for( const auto& component : basis::tensorProductComponentSplines( *out.patches.front().at( patch ) ) )
                counts.push_back( basis::numElements( component->knotVector() ) );
            eachIndex( counts, [&]( const Index& ix ) { out.leaves.front().at( patch ).insert( ix ); } );
        }
        std::mt19937 random( spec.seed );
        for( size_t level = 0; level < spec.ratios.size(); level++ )
            for( size_t patch = 0; patch < spec.patches; patch++ )
            {
                const auto candidates = out.leaves.at( level ).at( patch );
                for( const auto& ix : candidates )
                {
                    bool refine = spec.marks == "uniform";
                    if( spec.marks == "jump" ) refine = patch == 0 and ix.front() == ( level == 0 ? 1 : 2 );
                    if( spec.marks == "diagonal" ) refine = patch == 0 or patch == 3;
                    if( spec.marks == "random" ) refine = random() % 3 != 0;
                    if( not refine ) continue;
                    out.leaves.at( level ).at( patch ).erase( ix );
                    const Index& ratio = out.ratios.at( level ).at( patch );
                    eachIndex( ratio, [&]( const Index& child_offset ) {
                        Index child = child_offset;
                        for( size_t axis = 0; axis < child.size(); axis++ )
                            child.at( axis ) += ratio.at( axis ) * ix.at( axis );
                        out.leaves.at( level + 1 ).at( patch ).insert( child );
                    } );
                }
            }
        return out;
    }

    inline topology::Cell tensorCell( const topology::TPCombinatorialMap& cmap, const Index& index )
    {
        SmallVector<topology::Dart, 3> darts;
        for( const auto i : index ) darts.push_back( topology::Dart( i ) );
        return topology::Cell( topology::flattenFull( cmap, topology::FullyUnflattenedDart( darts ) ), cmap.dim() );
    }

    inline std::vector<std::vector<topology::Cell>> tensorLeaves( const FixtureLevels& fixture )
    {
        std::vector<std::vector<topology::Cell>> out( fixture.patches.size() );
        for( size_t level = 0; level < out.size(); level++ )
        {
            const auto& cmap = fixture.patches.at( level ).front()->basisComplex().parametricAtlas().cmap();
            for( const auto& ix : fixture.leaves.at( level ).front() )
                out.at( level ).push_back( tensorCell( cmap, ix ) );
        }
        return out;
    }

    inline std::vector<std::vector<topology::Cell>> multiLeaves(
        const FixtureLevels& fixture, const std::vector<std::shared_ptr<const basis::MultiPatchSplineSpace>>& levels )
    {
        std::vector<std::vector<topology::Cell>> out( levels.size() );
        for( size_t level = 0; level < levels.size(); level++ )
        {
            const auto& cmap = dynamic_cast<const topology::MultiPatchCombinatorialMap&>(
                levels.at( level )->basisComplex().parametricAtlas().cmap() );
            for( size_t patch = 0; patch < fixture.patches.at( level ).size(); patch++ )
                for( const auto& ix : fixture.leaves.at( level ).at( patch ) )
                    out.at( level ).emplace_back(
                        cmap.toGlobalDart( patch, tensorCell( *cmap.constituents().at( patch ), ix ).dart() ),
                        cmap.dim() );
        }
        return out;
    }
}
