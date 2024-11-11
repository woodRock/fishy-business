# Install script for directory: /home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX

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
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
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
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/OpenXLSX/headers" TYPE FILE FILES "/home/woodj/Desktop/fishy-business/build/OpenXLSX/OpenXLSX-Exports.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/OpenXLSX/headers" TYPE FILE FILES
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/IZipArchive.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLCell.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLCellIterator.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLCellRange.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLCellReference.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLCellValue.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLColor.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLColumn.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLCommandQuery.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLComments.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLConstants.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLContentTypes.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLDateTime.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLDocument.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLException.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLFormula.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLIterator.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLMergeCells.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLProperties.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLRelationships.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLRow.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLRowData.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLSharedStrings.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLSheet.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLStyles.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLWorkbook.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLXmlData.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLXmlFile.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLXmlParser.hpp"
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/headers/XLZipArchive.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/OpenXLSX" TYPE FILE FILES "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/OpenXLSX.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/woodj/Desktop/fishy-business/build/OpenXLSX/libOpenXLSXd.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX" TYPE FILE FILES
    "/home/woodj/Desktop/fishy-business/code/siamese/OpenXLSX/OpenXLSXConfig.cmake"
    "/home/woodj/Desktop/fishy-business/build/OpenXLSX/OpenXLSX/OpenXLSXConfigVersion.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX/OpenXLSXTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX/OpenXLSXTargets.cmake"
         "/home/woodj/Desktop/fishy-business/build/OpenXLSX/CMakeFiles/Export/lib/cmake/OpenXLSX/OpenXLSXTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX/OpenXLSXTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX/OpenXLSXTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX" TYPE FILE FILES "/home/woodj/Desktop/fishy-business/build/OpenXLSX/CMakeFiles/Export/lib/cmake/OpenXLSX/OpenXLSXTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX" TYPE FILE FILES "/home/woodj/Desktop/fishy-business/build/OpenXLSX/CMakeFiles/Export/lib/cmake/OpenXLSX/OpenXLSXTargets-debug.cmake")
  endif()
endif()

