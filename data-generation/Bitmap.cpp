#include "Bitmap.h"


myBitmap::myBitmap() : pen(NULL), brush(NULL), clr(0), wid(1) {}

myBitmap::~myBitmap() {
    DeleteObject(pen); DeleteObject(brush);
    DeleteDC(hdc); DeleteObject(bmp);
}

bool myBitmap::create(int w, int h) {
    BITMAPINFO bi;
    ZeroMemory(&bi, sizeof(bi));
    bi.bmiHeader.biSize = sizeof(bi.bmiHeader);
    bi.bmiHeader.biBitCount = sizeof(DWORD) * 8;
    bi.bmiHeader.biCompression = BI_RGB;
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biWidth = w;
    bi.bmiHeader.biHeight = -h;
    HDC dc = GetDC(GetConsoleWindow());
    bmp = CreateDIBSection(dc, &bi, DIB_RGB_COLORS, &pBits, NULL, 0);
    if (!bmp) return false;
    hdc = CreateCompatibleDC(dc);
    SelectObject(hdc, bmp);
    ReleaseDC(GetConsoleWindow(), dc);
    width = w; height = h;
    return true;
}

void myBitmap::clear(BYTE clr) {
    memset(pBits, clr, width * height * sizeof(DWORD));
}

void myBitmap::setBrushColor(DWORD bClr) {
    if (brush) DeleteObject(brush);
    brush = CreateSolidBrush(bClr);
    SelectObject(hdc, brush);
}

void myBitmap::setPenColor(DWORD c) {
    clr = c; createPen();
}

void myBitmap::setPenWidth(int w) {
    wid = w; createPen();
}

void myBitmap::saveBitmap(std::string path) {
    BITMAPFILEHEADER fileheader;
    BITMAPINFO       infoheader;
    BITMAP           bitmap;
    DWORD            wb;
    GetObject(bmp, sizeof(bitmap), &bitmap);
    DWORD* dwpBits = new DWORD[bitmap.bmWidth * bitmap.bmHeight];
    ZeroMemory(dwpBits, bitmap.bmWidth * bitmap.bmHeight * sizeof(DWORD));
    ZeroMemory(&infoheader, sizeof(BITMAPINFO));
    ZeroMemory(&fileheader, sizeof(BITMAPFILEHEADER));
    infoheader.bmiHeader.biBitCount = sizeof(DWORD) * 8;
    infoheader.bmiHeader.biCompression = BI_RGB;
    infoheader.bmiHeader.biPlanes = 1;
    infoheader.bmiHeader.biSize = sizeof(infoheader.bmiHeader);
    infoheader.bmiHeader.biHeight = bitmap.bmHeight;
    infoheader.bmiHeader.biWidth = bitmap.bmWidth;
    infoheader.bmiHeader.biSizeImage = bitmap.bmWidth * bitmap.bmHeight * sizeof(DWORD);
    fileheader.bfType = 0x4D42;
    fileheader.bfOffBits = sizeof(infoheader.bmiHeader) + sizeof(BITMAPFILEHEADER);
    fileheader.bfSize = fileheader.bfOffBits + infoheader.bmiHeader.biSizeImage;
    GetDIBits(hdc, bmp, 0, height, (LPVOID)dwpBits, &infoheader, DIB_RGB_COLORS);
    HANDLE file = CreateFile(path.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL, NULL);
    WriteFile(file, &fileheader, sizeof(BITMAPFILEHEADER), &wb, NULL);
    WriteFile(file, &infoheader.bmiHeader, sizeof(infoheader.bmiHeader), &wb, NULL);
    WriteFile(file, dwpBits, bitmap.bmWidth * bitmap.bmHeight * 4, &wb, NULL);
    CloseHandle(file);
    delete[] dwpBits;
}

HDC myBitmap::getDC() const { return hdc; }
int myBitmap::getWidth() const { return width; }
int myBitmap::getHeight() const { return height; }
DWORD* myBitmap::bits() const { return (DWORD*)pBits; }

void myBitmap::createPen() {
    if (pen) DeleteObject(pen);
    pen = CreatePen(PS_SOLID, wid, clr);
    SelectObject(hdc, pen);
}