/// @file    demo_wmark.cpp
/// @author  Martin Jirman (jirman@cesnet.cz)
/// @brief   2012 demo watermarker implementation.

#include <inttypes.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <wand/magick_wand.h>
#include "demo_wmark.h"
#include "cesnet-logo.h"


#define MAX_TEXT_LEN (4 * 1024)
#define FONT_SIZE 26

/// 2012 demo watermarker instance.
struct demo_wmark {
    // size of current watermark image
    int size_x, size_y;
    
    // precomputed data for current watermark image
    uint32_t * data;
    
    // text for current watermark image
    char text[MAX_TEXT_LEN + 1];
    
    // drawing stuff
    DrawingWand * font_wand;
    PixelWand * black_wand;
    PixelWand * transparent_wand;
    MagickWand * logo_wand;
};



/// @param scr_rgb  packed image RGB pixel
/// @param wm_rgb  packed watermark RGB premultiplied with alpha
/// @param wm_a  1024 - destination alpha
static uint32_t combine_10b(uint32_t src_rgb, uint32_t wm_rgb, int wm_a) {
    const uint64_t rb_mask = 0xFFC00FFC;
    const uint64_t g_mask = 0x3FF000;
    
    const uint64_t dest_rb  = (src_rgb & rb_mask) * wm_a;
    const uint32_t dest_g = (src_rgb & g_mask) * wm_a;
    const uint64_t dest_rgb = (dest_rb & (rb_mask << 10)) | (dest_g & (g_mask << 10));
    return (dest_rgb >> 10) + wm_rgb;
}



/// Creates 2012 demo watermarker.
demo_wmark * demo_wmark_create() {
    demo_wmark * const wmark = new demo_wmark;
    if(wmark) {
        // initialize current watermark to empty one
        wmark->data = 0;
        wmark->size_x = 0;
        wmark->size_y = 0;
        wmark->text[0] = 0;
        
        // alocate static drawing stuff
        wmark->font_wand = NewDrawingWand();
        wmark->black_wand = NewPixelWand();
        wmark->transparent_wand = NewPixelWand();
        
        // set draw colors and fonts
        PixelSetColor(wmark->transparent_wand, "none");
        PixelSetColor(wmark->black_wand, "black");
        DrawSetFillColor(wmark->font_wand, wmark->black_wand);
        DrawSetFont(wmark->font_wand, "DejaVu-Sans-Bold");
        DrawSetFontSize(wmark->font_wand, FONT_SIZE);
        DrawSetTextAntialias(wmark->font_wand, MagickTrue);
        
        // load logo
        wmark->logo_wand = NewMagickWand();
        MagickConstituteImage(
                wmark->logo_wand,
                cesnet_logo.width, 
                cesnet_logo.height,
                "RGBA", 
                CharPixel, 
                cesnet_logo.pixel_data
        );
    }
    return wmark;
}


/// Destroys 2012 demo watermarker.
void demo_wmark_destroy(demo_wmark * wmark) {
    // nothing to be done here yet
    if(wmark->data) {
        delete [] wmark->data;
        wmark->font_wand = DestroyDrawingWand(wmark->font_wand);
        wmark->black_wand = DestroyPixelWand(wmark->black_wand);
        wmark->transparent_wand = DestroyPixelWand(wmark->transparent_wand);
        wmark->logo_wand = DestroyMagickWand(wmark->logo_wand);
    }
    delete wmark;
}



/// Gets current time in milliseconds.
double get_time_ms() {
    timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec * 1000.0 + tv.tv_usec * 0.001;
}



/// Adds watermark to 10bit RGB image. 
/// @param wmark  watermarker instance
/// @param data  packed 10bit RGB dat ato be watermarked
/// @param text  text to be added as the watermark
/// @param size_x  width of the data
/// @param size_y  height of the data
void demo_wmark_add(demo_wmark * wmark, void * data, const char * text,
                    const int size_x, const int size_y) {
    // remember time of begin of watermarking
    const double begin_time_ms = get_time_ms();
    
    // recreate the watermark if have no watermark yet 
    // or if the text has changed
    if(0 == wmark->data || strcmp(wmark->text, text)) {
        // replace the text
        snprintf(wmark->text, MAX_TEXT_LEN, "%s", text);
        
        // remove old buffer if have one
        if(wmark->data) {
            delete [] wmark->data;
            wmark->data = 0;
        }
        
        // draw text into new transparent image
        DrawingWand * const font_wand = CloneDrawingWand(wmark->font_wand);
        MagickWand * const text_wand = NewMagickWand();
        MagickNewImage(text_wand, strlen(text) * FONT_SIZE, FONT_SIZE * 3,
                       wmark->transparent_wand);
        DrawAnnotation(font_wand, FONT_SIZE / 4, 2 * FONT_SIZE,
                       (const unsigned char*)text);
        MagickDrawImage(text_wand, font_wand);

        // trim the image down to include only the text and reset the origin
        MagickTrimImage(text_wand, 0);
        MagickResetImagePage(text_wand, "");
        
        // create an empty image with some border and copy the text into it
        const int shadow_size = FONT_SIZE / 3;
        const int text_sx = MagickGetImageWidth(text_wand);
        MagickWand * const final_wand = NewMagickWand();
        const int final_sx = text_sx + 2 * shadow_size + cesnet_logo.width;
        const int final_sy = cesnet_logo.height;
        MagickNewImage(final_wand, final_sx, final_sy, wmark->transparent_wand);
        
        // copy the text into the new image, invert it and blur it
        MagickCompositeImage(final_wand, text_wand, ReplaceCompositeOp,
                             shadow_size, cesnet_logo.height / 4 - 7);
        MagickNegateImage(final_wand, MagickTrue);
        MagickBlurImageChannel(final_wand, AlphaChannel, 0, shadow_size / 4);
        MagickGammaImageChannel(final_wand, AlphaChannel, 3.0);
        
        // add original text onto the blurred negative "shadow"
        MagickCompositeImage(final_wand, text_wand, OverCompositeOp,
                             shadow_size, cesnet_logo.height / 4 - 7);
        
        // add cesnet logo
        MagickCompositeImage(final_wand, wmark->logo_wand, OverCompositeOp,
                             text_sx + 2 * shadow_size, 0);
        
        // get raw data
        wmark->size_x = MagickGetImageWidth(final_wand);
        wmark->size_y = MagickGetImageHeight(final_wand);
        wmark->data = new uint32_t [wmark->size_x * wmark->size_y * 2];
        MagickExportImagePixels(final_wand, 0, 0, wmark->size_x, wmark->size_y,
                                "RGBA", CharPixel, wmark->data);
        
        // clean up
        DestroyMagickWand(text_wand);
        DestroyMagickWand(final_wand);
        DestroyDrawingWand(font_wand);
        
        // proprocess (in-place) the watermark for faster merging
        const int pix_count = wmark->size_x * wmark->size_y;
        uint32_t * dest = wmark->data + 2 * pix_count;
        const uint8_t * src = (uint8_t*)wmark->data + 4 * pix_count;
        for(int i = pix_count; i--;) {
            // update source pointer
            src -= 4;
            
            // watermark rgba (converted to 10bit)
            uint32_t a = (src[3] << 2) | (src[3] >> 6);
            uint32_t b = (src[2] << 2) | (src[2] >> 6);
            uint32_t g = (src[1] << 2) | (src[1] >> 6);
            uint32_t r = (src[0] << 2) | (src[0] >> 6);
            
            // write "one minus alpha"
            *(--dest) = 1024 - a;
            
            // premultiply and pack watermark rgb
            r = (r * a) >> 10;
            g = (g * a) >> 10;
            b = (b * a) >> 10;
            *(--dest) = (b << 02) | (g << 12) | (r << 22);
        }
    }
    
    // position of the watermark in the image
    const int end_y = size_y - 30;
    const int begin_y = end_y - wmark->size_y;
    const int end_x = size_x - 100;
    const int begin_x = end_x - wmark->size_x;
    
    // add the watermark
    const uint32_t * w = wmark->data;
    for(int y = begin_y; y < end_y; y++) {
        // line begin pointer
        uint32_t * p = (uint32_t*)data + begin_x + y * size_x;
        
        // for each pixel in the line
        for(int x = wmark->size_x; x--; ) {
            // combine with precomputed watermark if alpha not 0
            if(w[1] != 1024) {
                *p = combine_10b(*p, w[0], w[1]);
            }
            
            // advance pointers
            p += 1;
            w += 2;
        }
    }
    
    // show watermarking time
    const double end_time_ms = get_time_ms();
    printf("Watermarking time: %f ms.\n", end_time_ms - begin_time_ms);
}

