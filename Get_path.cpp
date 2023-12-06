#include <windows.h>
#include <iostream>
#include <string>

void FindFilesInDirectory(const std::wstring& directory) {
    WIN32_FIND_DATA fileData;
    std::wstring searchPath = directory + L"\\*.jpg"; // Path to .jpg files

    HANDLE hFind = FindFirstFile(searchPath.c_str(), &fileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            std::wstring fileName = fileData.cFileName;
            if ((fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
                std::wcout << L"File: " << directory << L"\\" << fileName << std::endl;
            }
        } while (FindNextFile(hFind, &fileData) != 0);
        FindClose(hFind);
    }
}

void FindFilesInDatabaseDirectory() {
    std::wstring databaseDirectory = L"Database"; // Path to the Database directory
    WIN32_FIND_DATA dirData;
    std::wstring searchDirPath = databaseDirectory + L"\\*";

    HANDLE hDir = FindFirstFile(searchDirPath.c_str(), &dirData);
    if (hDir != INVALID_HANDLE_VALUE) {
        do {
            if ((dirData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
                wcscmp(dirData.cFileName, L".") != 0 &&
                wcscmp(dirData.cFileName, L"..") != 0) {
                std::wstring subdirPath = databaseDirectory + L"\\" + dirData.cFileName;
                FindFilesInDirectory(subdirPath);
            }
        } while (FindNextFile(hDir, &dirData) != 0);
        FindClose(hDir);
    }
}

int main() {
    FindFilesInDatabaseDirectory();
    return 0;
}
