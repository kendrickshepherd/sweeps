#pragma once
#include <TPSplineSpace.hpp>
#include <CombinatorialMapMethods.hpp>
#include <SmallVector.hpp>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

// Internal to THB activity; not included by any public header.
namespace basis::detail
{
    /// @brief The element range [first, last) of each function of a B-spline space.
    /// Function i is supported on the unique-knot intervals spanned by its knots i, ..., i + degree + 1.
    class SupportRanges1d
    {
        public:
        /// @throws std::logic_error if the knot vector and the space's cmap disagree on the element count.
        explicit SupportRanges1d( const BSplineSpace1d& space )
        {
            const size_t degree = space.basisComplex().defaultParentBasis().mBasisGroups.at( 0 ).degrees.at( 0 );
            const auto& unique = space.knotVector().uniqueKnotMultiplicities();
            mNumElements = unique.size() - 1;
            if( mNumElements != topology::cellCount( space.basisComplex().parametricAtlas().cmap(), 1 ) )
                throw std::logic_error( "SupportRanges1d knot intervals differ from the space's elements" );

            std::vector<size_t> unique_index;
            for( size_t u = 0; u < unique.size(); u++ )
                unique_index.insert( unique_index.end(), unique.at( u ).second, u );
            for( size_t i = 0; i < space.numFunctions(); i++ )
                mRanges.emplace_back( unique_index.at( i ), unique_index.at( i + degree + 1 ) );
        }

        size_t numFunctions() const { return mRanges.size(); }
        size_t numElements() const { return mNumElements; }

        /// @throws std::out_of_range for a function outside the space.
        const std::pair<size_t, size_t>& range( const size_t fid ) const { return mRanges.at( fid ); }

        private:
        size_t mNumElements;
        std::vector<std::pair<size_t, size_t>> mRanges;
    };

    /// @brief The tensor cells in the support of each function of a B-spline tensor-product space.
    /// Cells are 1d element indices per axis, axis 0 first, as tensorCellSupports reports them.
    class TensorSupport
    {
        public:
        using Indices = SmallVector<size_t, 3>;
        using Ranges = SmallVector<std::pair<size_t, size_t>, 3>;
        /// Shares 1d tables between spaces with the same components; holds the components alive.
        using Cache = std::map<std::shared_ptr<const BSplineSpace1d>, std::shared_ptr<const SupportRanges1d>>;

        /// @throws std::invalid_argument if space is not a tensor product of BSplineSpace1d components.
        TensorSupport( const TPSplineSpace& space, Cache& cache ) { initialize( space, cache ); }

        explicit TensorSupport( const TPSplineSpace& space )
        {
            Cache cache;
            initialize( space, cache );
        }

        size_t numFunctions() const { return mNumFunctions; }

        /// @throws std::out_of_range for a function outside the space.
        Ranges ranges( const FunctionId& fid ) const
        {
            if( fid.id() < 0 or static_cast<size_t>( fid.id() ) >= mNumFunctions )
                throw std::out_of_range( "TensorSupport function is outside the space" );
            Ranges out;
            size_t rest = static_cast<size_t>( fid.id() );
            for( const auto& axis : mAxes )
            {
                out.push_back( axis->range( rest % axis->numFunctions() ) );
                rest /= axis->numFunctions();
            }
            return out;
        }

        /// @brief Calls callback( const Indices& ) on each support cell, axis 0 fastest, until it returns false.
        /// @return false iff the callback stopped the iteration.
        /// @throws std::out_of_range for a function outside the space.
        template <typename Callback>
        bool iterateCellsWhile( const FunctionId& fid, Callback&& callback ) const
        {
            const Ranges r = ranges( fid );
            Indices cell;
            for( const auto& axis : r ) cell.push_back( axis.first );
            while( true )
            {
                if( not callback( static_cast<const Indices&>( cell ) ) ) return false;
                size_t axis = 0;
                while( axis < r.size() and ++cell.at( axis ) == r.at( axis ).second )
                {
                    cell.at( axis ) = r.at( axis ).first;
                    axis++;
                }
                if( axis == r.size() ) return true;
            }
        }

        private:
        void initialize( const TPSplineSpace& space, Cache& cache )
        {
            const auto components = tensorProductComponentSplines( space );
            if( components.empty() ) throw std::invalid_argument( "TensorSupport requires B-spline components" );
            mNumFunctions = 1;
            for( const auto& component : components )
            {
                auto it = cache.find( component );
                if( it == cache.end() )
                    it = cache.emplace( component, std::make_shared<const SupportRanges1d>( *component ) ).first;
                mAxes.push_back( it->second );
                mNumFunctions *= it->second->numFunctions();
            }
        }

        SmallVector<std::shared_ptr<const SupportRanges1d>, 3> mAxes;
        size_t mNumFunctions;
    };
}
