#pragma once

#include <windows.h>
#include <string>

class myBitmap {
public:
    myBitmap();
    ~myBitmap();

    bool create(int w, int h);
    void clear(BYTE clr = 0);
    void setBrushColor(DWORD bClr);
    void setPenColor(DWORD c);
    void setPenWidth(int w);
    void saveBitmap(std::string path);
    HDC getDC() const;
    int getWidth() const;
    int getHeight() const;
    DWORD* bits() const;
private:
    void createPen();
    HBITMAP bmp; HDC    hdc;
    HPEN    pen; HBRUSH brush;
    void* pBits; int    width, height, wid;
    DWORD    clr;
};