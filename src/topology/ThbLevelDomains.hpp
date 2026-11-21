#pragma once
#include <HierarchicalMultiPatchCombinatorialMap.hpp>
#include <HierarchicalTPCombinatorialMap.hpp>
#include <CombinatorialMapMethods.hpp>
#include <TPCombinatorialMap.hpp>
#include <SmallVector.hpp>
#include <algorithm>
#include <stdexcept>
#include <vector>

// Internal to THB activity; not included by any public header.
namespace topology::detail
{
    /// @brief Membership of level cells in Omega^level, from the leaves alone.
    /// A level cell is outside Omega^level iff a coarser ancestor is a leaf.  Ancestor indices are the
    /// cell's tensor indices divided by the patch's refinement ratios, as iterateChildren assumes.
    /// Leaves are trusted to partition each patch, as the hierarchical cmaps guarantee.
    class LevelDomains
    {
        public:
        using Indices = SmallVector<size_t, 3>;

        struct PatchLevels
        {
            /// Level-0 tensor cell counts per axis.
            Indices base_lengths;
            /// Per transition, the refinement ratio per axis, as refinementRatios().
            std::vector<Indices> ratios;
        };

        struct Leaf
        {
            size_t patch;
            size_t level;
            Indices indices;
        };

        /// @throws std::invalid_argument for no patches, mixed dimensions, a zero length or ratio,
        /// or a leaf outside its patch's levels or cells.
        LevelDomains( const std::vector<PatchLevels>& patches, const std::vector<Leaf>& leaves )
        {
            if( patches.empty() ) throw std::invalid_argument( "LevelDomains requires at least one patch" );
            const size_t dim = patches.front().base_lengths.size();
            if( dim == 0 ) throw std::invalid_argument( "LevelDomains requires a positive dimension" );
            for( const PatchLevels& patch : patches )
            {
                if( patch.base_lengths.size() != dim )
                    throw std::invalid_argument( "LevelDomains patches must share a dimension" );
                for( const Indices& ratio : patch.ratios )
                    if( ratio.size() != dim )
                        throw std::invalid_argument( "LevelDomains ratio has the wrong dimension" );
                const auto zero = []( const Indices& v ) { return std::find( v.begin(), v.end(), 0 ) != v.end(); };
                if( zero( patch.base_lengths ) or std::any_of( patch.ratios.begin(), patch.ratios.end(), zero ) )
                    throw std::invalid_argument( "LevelDomains lengths and ratios must be positive" );

                std::vector<Indices>& lengths = mLengths.emplace_back( 1, patch.base_lengths );
                for( const Indices& ratio : patch.ratios )
                {
                    Indices next = lengths.back();
                    for( size_t axis = 0; axis < dim; axis++ ) next.at( axis ) *= ratio.at( axis );
                    lengths.push_back( next );
                }
                mRatios.push_back( patch.ratios );
                mLeaves.emplace_back( lengths.size() );
            }

            for( const Leaf& leaf : leaves )
            {
                if( leaf.patch >= mLengths.size() or leaf.level >= mLengths.at( leaf.patch ).size() or
                    not inBounds( mLengths.at( leaf.patch ).at( leaf.level ), leaf.indices ) )
                    throw std::invalid_argument( "LevelDomains leaf is outside the hierarchy" );
                mLeaves.at( leaf.patch ).at( leaf.level ).push_back(
                    linear( mLengths.at( leaf.patch ).at( leaf.level ), leaf.indices ) );
            }
            for( auto& patch : mLeaves )
                for( auto& level : patch ) std::sort( level.begin(), level.end() );
        }

        size_t numPatches() const { return mLengths.size(); }
        size_t numLevels( const size_t patch ) const { return mLengths.at( patch ).size(); }
        const Indices& lengths( const size_t patch, const size_t level ) const
        {
            return mLengths.at( patch ).at( level );
        }

        /// @brief Whether a level cell lies in Omega^level.
        /// @throws std::out_of_range for a patch, level or cell outside the hierarchy.
        bool contains( const size_t patch, const size_t level, const Indices& cell ) const
        {
            if( patch >= mLengths.size() or level >= mLengths.at( patch ).size() or
                not inBounds( mLengths.at( patch ).at( level ), cell ) )
                throw std::out_of_range( "LevelDomains query is outside the hierarchy" );

            Indices ancestor = cell;
            for( size_t coarse = level; coarse-- > 0; )
            {
                const Indices& ratio = mRatios.at( patch ).at( coarse );
                for( size_t axis = 0; axis < ancestor.size(); axis++ ) ancestor.at( axis ) /= ratio.at( axis );
                const std::vector<size_t>& leaves = mLeaves.at( patch ).at( coarse );
                if( std::binary_search( leaves.begin(), leaves.end(),
                                        linear( mLengths.at( patch ).at( coarse ), ancestor ) ) )
                    return false;
            }
            return true;
        }

        private:
        static bool inBounds( const Indices& lengths, const Indices& cell )
        {
            if( cell.size() != lengths.size() ) return false;
            for( size_t axis = 0; axis < cell.size(); axis++ )
                if( cell.at( axis ) >= lengths.at( axis ) ) return false;
            return true;
        }

        static size_t linear( const Indices& lengths, const Indices& cell )
        {
            size_t out = 0;
            for( size_t axis = cell.size(); axis-- > 0; ) out = out * lengths.at( axis ) + cell.at( axis );
            return out;
        }

        std::vector<std::vector<Indices>> mLengths;               // [patch][level]
        std::vector<std::vector<Indices>> mRatios;                // [patch][transition]
        std::vector<std::vector<std::vector<size_t>>> mLeaves;    // [patch][level], sorted linear indices
    };

    /// @brief The 1d dart ids of a top-dimensional tensor cell, as leafElementInfo reports them.
    inline LevelDomains::Indices tensorCellIndices( const TPCombinatorialMap& cmap, const Cell& cell )
    {
        if( cell.dim() != cmap.dim() )
            throw std::invalid_argument( "tensorCellIndices requires a top-dimensional cell" );
        LevelDomains::Indices out;
        for( const Dart& d : unflattenFull( cmap, cell.dart() ).unflat_darts ) out.push_back( d.id() );
        return out;
    }

    inline LevelDomains::Indices tensorCellLengths( const TPCombinatorialMap& cmap )
    {
        LevelDomains::Indices out;
        for( const auto& component : tensorProductComponentCMaps( cmap ) ) out.push_back( cellCount( *component, 1 ) );
        return out;
    }

    /// @param leaf_elements  Level-local leaf cells per level, as levelActiveFunctions receives them.
    /// @throws std::invalid_argument unless there is one leaf list per level of top-dimensional cells.
    inline LevelDomains levelDomains( const HierarchicalTPCombinatorialMap& cmap,
                                      const std::vector<std::vector<Cell>>& leaf_elements )
    {
        if( leaf_elements.size() != cmap.numLevels() )
            throw std::invalid_argument( "levelDomains requires one leaf list per level" );
        std::vector<LevelDomains::Leaf> leaves;
        for( size_t level = 0; level < leaf_elements.size(); level++ )
            for( const Cell& leaf : leaf_elements.at( level ) )
                leaves.push_back( { 0, level, tensorCellIndices( *cmap.refinementLevels().at( level ), leaf ) } );
        return LevelDomains( { { tensorCellLengths( *cmap.refinementLevels().front() ), cmap.refinementRatios() } },
                             leaves );
    }

    /// @param leaf_elements  Level-local leaf cells per level, as levelActiveFunctions receives them.
    /// @throws std::invalid_argument unless there is one leaf list per level of top-dimensional cells.
    inline LevelDomains levelDomains( const HierarchicalMultiPatchCombinatorialMap& cmap,
                                      const std::vector<std::vector<Cell>>& leaf_elements )
    {
        if( leaf_elements.size() != cmap.refinementLevels().size() )
            throw std::invalid_argument( "levelDomains requires one leaf list per level" );
        std::vector<LevelDomains::PatchLevels> patches;
        for( const auto& constituent : cmap.constituents() )
            patches.push_back( { tensorCellLengths( *constituent->refinementLevels().front() ),
                                 constituent->refinementRatios() } );
        std::vector<LevelDomains::Leaf> leaves;
        for( size_t level = 0; level < leaf_elements.size(); level++ )
        {
            const MultiPatchCombinatorialMap& level_cmap = *cmap.refinementLevels().at( level );
            for( const Cell& leaf : leaf_elements.at( level ) )
            {
                if( leaf.dim() != level_cmap.dim() )
                    throw std::invalid_argument( "tensorCellIndices requires a top-dimensional cell" );
                const auto [patch, local] = level_cmap.toLocalDart( leaf.dart() );
                const Cell local_cell( local, leaf.dim() );
                leaves.push_back(
                    { patch, level, tensorCellIndices( *level_cmap.constituents().at( patch ), local_cell ) } );
            }
        }
        return LevelDomains( patches, leaves );
    }
}
