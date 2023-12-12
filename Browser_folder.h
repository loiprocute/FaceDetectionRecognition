#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <tuple>


// Function to find files in a directory and its subdirectories
std::tuple<std::vector<std::string>, std::vector<std::string>> FindFilesInDirectory(const std::string& directory, const std::string& subdir, const std::string& format);
// Function to find files in the Database directory and its subdirectories
std::tuple<std::vector<std::string>, std::vector<std::string>> FindFilesInDatabaseDirectory(const std::string& format);

