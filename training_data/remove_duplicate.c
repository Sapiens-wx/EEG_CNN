#include <stdio.h>
#include <string.h>
#include <windows.h>
#include <tchar.h>
#define BUF_LEN 1000

int removeFile(const char* filename) {
    // 转换 const char* 到 LPCWSTR（宽字符）
    int wlen = MultiByteToWideChar(CP_UTF8, 0, filename, -1, NULL, 0);
    if (wlen == 0) {
        fprintf(stderr, "文件名转换失败 (MultiByteToWideChar)\n");
        return -1;
    }

    wchar_t* wfilename = (wchar_t*)malloc(wlen * sizeof(wchar_t));
    if (wfilename == NULL) {
        fprintf(stderr, "内存分配失败\n");
        return -1;
    }

    if (MultiByteToWideChar(CP_UTF8, 0, filename, -1, wfilename, wlen) == 0) {
        fprintf(stderr, "文件名转换失败 (MultiByteToWideChar)\n");
        free(wfilename);
        return -1;
    }

    // 尝试删除文件
    if (!DeleteFileW(wfilename)) {
        DWORD error = GetLastError();
        fprintf(stderr, "删除文件失败 (错误代码: %lu)\n", error);
        free(wfilename);
        return -1;
    }

    free(wfilename);
    return 0; // 成功
}

// 重命名文件
int renameFile(const char* newname, const char* oldname) {
    // 转换旧文件名
    int wlen_old = MultiByteToWideChar(CP_UTF8, 0, oldname, -1, NULL, 0);
    if (wlen_old == 0) {
        fprintf(stderr, "旧文件名转换失败\n");
        return -1;
    }

    wchar_t* woldname = (wchar_t*)malloc(wlen_old * sizeof(wchar_t));
    if (woldname == NULL) {
        fprintf(stderr, "内存分配失败\n");
        return -1;
    }

    if (MultiByteToWideChar(CP_UTF8, 0, oldname, -1, woldname, wlen_old) == 0) {
        fprintf(stderr, "旧文件名转换失败\n");
        free(woldname);
        return -1;
    }

    // 转换新文件名
    int wlen_new = MultiByteToWideChar(CP_UTF8, 0, newname, -1, NULL, 0);
    if (wlen_new == 0) {
        fprintf(stderr, "新文件名转换失败\n");
        free(woldname);
        return -1;
    }

    wchar_t* wnewname = (wchar_t*)malloc(wlen_new * sizeof(wchar_t));
    if (wnewname == NULL) {
        fprintf(stderr, "内存分配失败\n");
        free(woldname);
        return -1;
    }

    if (MultiByteToWideChar(CP_UTF8, 0, newname, -1, wnewname, wlen_new) == 0) {
        fprintf(stderr, "新文件名转换失败\n");
        free(woldname);
        free(wnewname);
        return -1;
    }

    // 尝试重命名文件
    if (!MoveFileW(woldname, wnewname)) {
        DWORD error = GetLastError();
        fprintf(stderr, "重命名文件失败 (错误代码: %lu)\n", error);
        free(woldname);
        free(wnewname);
        return -1;
    }

    free(woldname);
    free(wnewname);
    return 0; // 成功
}

void remove_duplicate(const char* path){
	FILE* infile=fopen(path, "r");
	int pathLength=strlen(path);
	char outfile_name[pathLength+5];
	//init output file path
	strcpy(outfile_name, path);
	strcat(outfile_name+pathLength, ".tmp");
	//open output file
	FILE* outfile=fopen(outfile_name, "w");
	char buffer[BUF_LEN];
	//copy first line and remove the return character
	fgets(buffer, BUF_LEN, infile);
	for(int i=strlen(buffer)-1;i>-1&&(buffer[i]=='\n'||buffer[i]=='\r');i--){
		buffer[i]=0;
	}
	fwrite(buffer, strlen(buffer), 1, outfile);
	
	double laststamp=0;
	int removed_cnt=0, total=0;
	while(!feof(infile)){
		double timestamp;
		fscanf(infile, "%lf,%s", &timestamp, buffer);
		if(timestamp>laststamp){
			fprintf(outfile, "\n%.3lf,%s", timestamp, buffer);
			laststamp=timestamp;
		} else{
			++removed_cnt;
		}
		++total;
	}
	printf("%s: %d/%d lines removed\n",path,removed_cnt,total);
	fclose(infile);
	fclose(outfile);
	removeFile(path);
	renameFile(path, outfile_name);
}

void ls() {
    WIN32_FIND_DATA fileData;
    HANDLE hFind;

    hFind = FindFirstFile(_T("*.csv"), &fileData);

    if (hFind == INVALID_HANDLE_VALUE) {
        printf("cannot open directory: %d\n", GetLastError());
        return;
    }

    do {
	//skip . and ..
        if (_tcscmp(fileData.cFileName, _T(".")) == 0 || _tcscmp(fileData.cFileName, _T("..")) == 0) {
            continue;
        }

        // 打印文件名
	remove_duplicate(fileData.cFileName);
    } while (FindNextFile(hFind, &fileData));

    // 关闭句柄
    FindClose(hFind);
}

int main(int argc, char** argv){
	if(argc==1)
		ls();
	else{
		for(int i=1;i<argc;++i){
			remove_duplicate(argv[i]);
		}
	}
	return 0;
}
