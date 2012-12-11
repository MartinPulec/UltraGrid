/**
 * @file    demo_wmark.h
 * @author  Martin Jirman (jirman@cesnet.cz)
 * @brief   2012 demo watermarker.
 */

#ifndef DEMO_WMARK_H
#define DEMO_WMARK_H

#ifdef __cplusplus
extern "C" {
#endif


/** 2012 demo watermarker. */
struct demo_wmark;


/** Creates 2012 demo watermarker. */
struct demo_wmark * demo_wmark_create();


/** Destroys 2012 demo watermarker. */
void demo_wmark_destroy(struct demo_wmark * wmark);


/** 
 * Adds watermark to 10bit RGB image. 
 * @param wmark  watermarker instance
 * @param data  packed 10bit RGB dat ato be watermarked
 * @param text  text to be added as the watermark
 * @param size_x  width of the data
 * @param size_y  height of the data
 */
void demo_wmark_add(struct demo_wmark * wmark, void * data, const char * text,
                    const int size_x, const int size_y);


#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* DEMO_WMARK_H */
