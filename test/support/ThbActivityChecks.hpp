#pragma once
#include "ThbActivityReference.hpp"
#include <ThbActiveFunctions.hpp>
#include <HierarchicalMultiPatchSplineSpace.hpp>
#include <VectorConformingHierarchicalMultiPatchSplineSpace.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>

namespace thb_test
{
    struct ActivityReport
    {
        size_t levels = 0;
        double reference_seconds = 0.0;
        double production_seconds = 0.0;
        // Stable summary only, not a substitute for the exact comparisons below.
        uint64_t fingerprint = 14695981039346656037ULL;
    };

    inline void hashWord( ActivityReport& report, uint64_t word )
    {
        for( size_t byte = 0; byte < 8; byte++, word >>= 8 )
        {
            report.fingerprint ^= word & 255;
            report.fingerprint *= 1099511628211ULL;
        }
    }

    template <typename CMap, typename Levels>
    void compareLevels( const CMap& cmap, const Levels& levels,
                        const std::vector<std::set<basis::FunctionId>>& stored,
                        const std::string& name, ActivityReport& report )
    {
        if( stored.size() != levels.size() ) throw std::runtime_error( name + ": wrong stored level count" );
        const auto leaves = leafElements( cmap );
        for( const unsigned char ch : name ) hashWord( report, ch );
        for( size_t level = 0; level < levels.size(); level++ )
        {
            const auto start = std::chrono::steady_clock::now();
            const auto expected = referenceLevelActiveFunctions( cmap, leaves, level, *levels.at( level ) );
            const auto middle = std::chrono::steady_clock::now();
            const auto actual = basis::detail::levelActiveFunctions( cmap, leaves, level, *levels.at( level ) );
            const auto end = std::chrono::steady_clock::now();
            report.reference_seconds += std::chrono::duration<double>( middle - start ).count();
            report.production_seconds += std::chrono::duration<double>( end - middle ).count();
            const std::string where = name + " level " + std::to_string( level );
            requireSameActiveFunctions( expected, actual, where + " direct" );
            requireSameActiveFunctions( expected, stored.at( level ), where + " constructor" );
            hashWord( report, level );
            hashWord( report, expected.size() );
            for( const auto& fid : expected ) hashWord( report, fid.id() );
            report.levels++;
        }
    }

    inline void checkActivity( const basis::HierarchicalTPSplineSpace& ss, const std::string& name,
                               ActivityReport& report )
    {
        std::vector<std::set<basis::FunctionId>> stored;
        for( const auto& ids : ss.activeFunctions() )
        {
            if( not std::is_sorted( ids.begin(), ids.end() ) or
                std::adjacent_find( ids.begin(), ids.end() ) != ids.end() )
                throw std::runtime_error( name + ": stored active ids are not strictly increasing" );
            stored.emplace_back( ids.begin(), ids.end() );
        }
        compareLevels( ss.basisComplex().parametricAtlas().cmap(), ss.refinementLevels(), stored, name, report );
    }

    inline std::vector<std::set<basis::FunctionId>> storedGlobalSets(
        const std::vector<std::pair<size_t, basis::FunctionId>>& mapping, const size_t num_levels )
    {
        std::vector<std::set<basis::FunctionId>> stored( num_levels );
        for( const auto& [level, fid] : mapping )
            if( not stored.at( level ).insert( fid ).second )
                throw std::runtime_error( "Duplicate stored hierarchical (level, function) pair" );
        return stored;
    }

    inline void checkActivity( const basis::HierarchicalMultiPatchSplineSpace& ss, const std::string& name,
                               ActivityReport& report )
    {
        const auto mapping = basis::fidsToRefinementLevelFids( ss );
        if( mapping.size() != ss.numFunctions() ) throw std::runtime_error( name + ": wrong stored mapping size" );
        compareLevels( ss.basisComplex().parametricAtlas().cmap(), ss.refinementLevels(),
                       storedGlobalSets( mapping, ss.refinementLevels().size() ), name, report );
    }

    inline void checkActivity( const basis::VectorConformingHierarchicalMultiPatchSplineSpace& ss,
                               const std::string& name, ActivityReport& report )
    {
        const auto mapping = basis::vectorFidsToRefinementLevelFids( ss );
        if( mapping.size() != ss.numFunctions() ) throw std::runtime_error( name + ": wrong stored mapping size" );
        const auto& cmap = dynamic_cast<const topology::HierarchicalMultiPatchCombinatorialMap&>(
            ss.basisComplex().parametricAtlas().cmap() );
        compareLevels( cmap, ss.refinementLevels(), storedGlobalSets( mapping, ss.refinementLevels().size() ),
                       name, report );
    }

    inline void checkActivity( const basis::VectorConformingHierarchicalTPSplineSpace& ss,
                               const std::string& name, ActivityReport& report )
    {
        for( size_t component = 0; component < ss.scalarBases().size(); component++ )
            checkActivity( *ss.scalarBases().at( component ),
                           name + " component " + std::to_string( component ), report );
    }
}
