#include <windows.h>
#include <string>
#include <complex>
#include <ctime>  
#include <chrono>

const int BMP_SIZE = 2000, ITERATIONS = 1024;
const long double xmin = -2, xmax = 2;
const long double xwidth = xmax - xmin;
const long double ymin = -2, ymax = 2;
const long double yheight = ymax - ymin;

class myBitmap {
public:
    myBitmap() : pen(NULL), brush(NULL), clr(0), wid(1) {}
    ~myBitmap() {
        DeleteObject(pen); DeleteObject(brush);
        DeleteDC(hdc); DeleteObject(bmp);
    }
    bool create(int w, int h) {
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
    void clear(BYTE clr = 0) {
        memset(pBits, clr, width * height * sizeof(DWORD));
    }
    void setBrushColor(DWORD bClr) {
        if (brush) DeleteObject(brush);
        brush = CreateSolidBrush(bClr);
        SelectObject(hdc, brush);
    }
    void setPenColor(DWORD c) {
        clr = c; createPen();
    }
    void setPenWidth(int w) {
        wid = w; createPen();
    }
    void saveBitmap(std::string path) {
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
    HDC getDC() const { return hdc; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    DWORD* bits() const { return (DWORD*)pBits; }
private:
    void createPen() {
        if (pen) DeleteObject(pen);
        pen = CreatePen(PS_SOLID, wid, clr);
        SelectObject(hdc, pen);
    }
    HBITMAP bmp; HDC    hdc;
    HPEN    pen; HBRUSH brush;
    void* pBits; int    width, height, wid;
    DWORD    clr;
};
class julia {
public:
    void draw(std::complex<long double> k) {
        bmp.create(BMP_SIZE, BMP_SIZE);
        DWORD* bits = bmp.bits();
        int res, pos;
        std::complex<long double> c;

        for (int y = 0; y < BMP_SIZE; y++) {
            pos = y * BMP_SIZE;
            for (int x = 0; x < BMP_SIZE; x++) {
                c.imag((double)y / (double)BMP_SIZE * yheight + ymin);
                c.real((double)x / (double)BMP_SIZE * xwidth + xmin);
                res = inSet(c, k);
                if (res) {
                    int n_res = res % 255;
                    if (res < (ITERATIONS >> 1)) res = RGB(n_res << 2, n_res << 3, n_res << 4);
                    else res = RGB(n_res << 4, n_res << 2, n_res << 5);
                }
                bits[pos++] = res;
            }
        }
        bmp.saveBitmap("./js.bmp");
    }
private:
    int inSet(std::complex<long double> z, std::complex<long double> c) {
        long double dist;//, three = 3.0;
        for (int ec = 0; ec < ITERATIONS; ec++) {
            z = z * z; z = z + c;
            dist = (z.imag() * z.imag()) + (z.real() * z.real());
            if (dist > 10) return(ec);
        }
        return 0;
    }
    myBitmap bmp;
};
int main(int argc, char* argv[]) {
    std::complex<long double> c;
    c.imag(0.2733);
    c.real(0.27015);
    auto start = std::chrono::system_clock::now();
    julia j; j.draw(c); 
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    printf("%f", elapsed_seconds.count());
    return 0;
}