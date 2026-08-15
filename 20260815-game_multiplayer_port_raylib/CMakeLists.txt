cmake_minimum_required(VERSION 3.20)
project(broadside VERSION 1.0 LANGUAGES C CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-ffp-contract=off)
endif()
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BROADSIDE_DESKTOP "Build the raylib desktop client" ON)
option(BROADSIDE_ENET "Build the incomplete ENet transport adapter (not used by the runtime)" OFF)
option(BROADSIDE_FETCH_DEPS "Fetch pinned third-party dependencies" ON)
option(BROADSIDE_SANITIZERS "Enable address and undefined-behaviour sanitizers" OFF)

include(FetchContent)

if(BROADSIDE_DESKTOP AND BROADSIDE_FETCH_DEPS)
  FetchContent_Declare(raylib
    URL https://github.com/raysan5/raylib/archive/dbc56a87da87d973a9c5baa4e7438a9d20121d28.tar.gz)
  FetchContent_MakeAvailable(raylib)
endif()

if(BROADSIDE_ENET AND BROADSIDE_FETCH_DEPS)
  FetchContent_Declare(enet
    URL https://github.com/zpl-c/enet/archive/8b43b92591fa9458662262a598b092ecc5132865.tar.gz)
  FetchContent_MakeAvailable(enet)
endif()

add_library(broadside_sim
  native/sim/rng.cpp
  native/sim/geometry.cpp
  native/sim/data.cpp
  native/sim/ship.cpp
  native/sim/battle.cpp
  native/sim/timeline.cpp
  native/sim/checksum.cpp
)
target_include_directories(broadside_sim PUBLIC native)
target_compile_features(broadside_sim PUBLIC cxx_std_20)

add_library(broadside_game
  native/game/shipyard.cpp
  native/game/match.cpp
  native/game/autobuild.cpp
  native/game/bot.cpp
)
target_link_libraries(broadside_game PUBLIC broadside_sim)
target_include_directories(broadside_game PUBLIC native)

add_library(broadside_net
  native/net/protocol.cpp
  native/net/authority.cpp
  native/net/transport.cpp
  native/net/client.cpp
  native/net/session.cpp
)
target_link_libraries(broadside_net PUBLIC broadside_game)
target_include_directories(broadside_net PUBLIC native)
if(BROADSIDE_ENET)
  target_sources(broadside_net PRIVATE native/net/enet_transport.cpp)
endif()

add_library(broadside_audio
  native/audio/audio.cpp
)
target_include_directories(broadside_audio PUBLIC native)

add_library(broadside_reference native/reference/reference.cpp)
target_include_directories(broadside_reference PUBLIC native)

add_library(broadside_core INTERFACE)
target_link_libraries(broadside_core INTERFACE broadside_net broadside_audio broadside_reference)

add_library(broadside_presentation
  native/presentation/battle_visual.cpp
  native/presentation/controller.cpp
  native/presentation/quality.cpp
)
target_include_directories(broadside_presentation PUBLIC native)
target_link_libraries(broadside_presentation PUBLIC broadside_net)
target_compile_options(broadside_presentation PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>)
if(BROADSIDE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(broadside_presentation PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(broadside_presentation PUBLIC -fsanitize=address,undefined)
endif()

foreach(target broadside_sim broadside_game broadside_net broadside_audio broadside_reference)
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_compile_options(${target} PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>)
endforeach()

if(BROADSIDE_ENET AND TARGET enet)
  target_compile_definitions(broadside_net PUBLIC BROADSIDE_HAS_ENET=1)
  target_link_libraries(broadside_net PUBLIC enet)
endif()

if(BROADSIDE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  foreach(target broadside_sim broadside_game broadside_net broadside_audio broadside_reference)
    target_compile_options(${target} PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PUBLIC -fsanitize=address,undefined)
  endforeach()
endif()

add_executable(broadside_tool native/tools/main.cpp)
target_link_libraries(broadside_tool PRIVATE broadside_core broadside_presentation)
target_compile_definitions(broadside_tool PRIVATE BROADSIDE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

if(BROADSIDE_DESKTOP)
  add_executable(broadside native/main.cpp native/render/renderer.cpp native/render/ui.cpp)
  target_link_libraries(broadside PRIVATE broadside_core broadside_presentation)
  target_compile_options(broadside PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>)
  if(TARGET raylib)
    target_link_libraries(broadside PRIVATE raylib)
    target_compile_definitions(broadside PRIVATE BROADSIDE_HAS_RAYLIB=1)
    target_compile_definitions(broadside PRIVATE BROADSIDE_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/native/assets")
    add_custom_command(TARGET broadside POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/native/assets" "$<TARGET_FILE_DIR:broadside>/assets")
  else()
    message(WARNING "raylib was not made available; broadside desktop target will be omitted")
    set_target_properties(broadside PROPERTIES EXCLUDE_FROM_ALL TRUE EXCLUDE_FROM_DEFAULT_BUILD TRUE)
  endif()
endif()

enable_testing()
add_test(NAME broadside_invariants COMMAND broadside_tool test)
add_test(NAME broadside_golden_repeatability COMMAND broadside_tool golden)
add_test(NAME broadside_reference_fixture COMMAND broadside_tool reference)
add_test(NAME broadside_network_replay COMMAND broadside_tool netcheck)
add_test(NAME broadside_audio_envelope COMMAND broadside_tool audio-check)
add_test(NAME broadside_playthrough COMMAND broadside_tool playthrough)
add_test(NAME broadside_render_layout COMMAND broadside_tool rendercheck)
add_test(NAME broadside_melee_envelope COMMAND broadside_tool melee)
set_tests_properties(broadside_invariants PROPERTIES LABELS "quick;unit")
set_tests_properties(broadside_golden_repeatability broadside_reference_fixture PROPERTIES LABELS "determinism;reference")
set_tests_properties(broadside_network_replay PROPERTIES LABELS "network")
set_tests_properties(broadside_audio_envelope PROPERTIES LABELS "quick;audio")
set_tests_properties(broadside_playthrough PROPERTIES LABELS "gameplay")
set_tests_properties(broadside_render_layout PROPERTIES LABELS "presentation")
set_tests_properties(broadside_melee_envelope PROPERTIES LABELS "balance;reference")
