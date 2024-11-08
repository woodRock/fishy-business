# Install script for directory: /vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/OpenXLSX/headers" TYPE FILE FILES "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/dev/OpenXLSX/OpenXLSX-Exports.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/OpenXLSX/headers" TYPE FILE FILES
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/IZipArchive.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLCell.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLCellIterator.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLCellRange.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLCellReference.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLCellValue.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLColor.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLColumn.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLCommandQuery.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLComments.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLConstants.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLContentTypes.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLDateTime.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLDocument.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLException.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLFormula.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLIterator.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLMergeCells.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLProperties.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLRelationships.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLRow.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLRowData.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLSharedStrings.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLSheet.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLStyles.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLWorkbook.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLXmlData.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLXmlFile.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLXmlParser.hpp"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/headers/XLZipArchive.hpp"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/OpenXLSX" TYPE FILE FILES "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/OpenXLSX.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "lib" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/dev/OpenXLSX/libOpenXLSX.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX" TYPE FILE FILES
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/OpenXLSX/OpenXLSXConfig.cmake"
    "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/dev/OpenXLSX/OpenXLSX/OpenXLSXConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX/OpenXLSXTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX/OpenXLSXTargets.cmake"
         "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/dev/OpenXLSX/CMakeFiles/Export/c72cc94553a1a0c9b05f75dae42fb1d7/OpenXLSXTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX/OpenXLSXTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX/OpenXLSXTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX" TYPE FILE FILES "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/dev/OpenXLSX/CMakeFiles/Export/c72cc94553a1a0c9b05f75dae42fb1d7/OpenXLSXTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX" TYPE FILE FILES "/vol/ecrg-solar/woodj4/fishy-business/code/siamese/dev/OpenXLSX/CMakeFiles/Export/c72cc94553a1a0c9b05f75dae42fb1d7/OpenXLSXTargets-noconfig.cmake")
  endif()
endif()

