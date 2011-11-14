#version 130

#extension GL_EXT_gpu_shader4 : enable

#define lerp mix

in vec4 TEX0;
uniform sampler2D image;
uniform float imageWidth;
out vec4 color;

void main()
{
        vec4 yuv;
        yuv.rgba  = texture2D(image, gl_TexCoord[0].xy).grba;
        if(gl_TexCoord[0].x * imageWidth / 2.0 - floor(gl_TexCoord[0].x * imageWidth / 2.0) > 0.5)
                yuv.r = yuv.a;
        yuv.r = 1.1643*(yuv.r-0.0625);
        yuv.g = yuv.g - 0.5;
        yuv.b = yuv.b - 0.5;
        color.r = yuv.r + 1.5958 * yuv.b;
        color.g = yuv.r - 0.39173* yuv.g - 0.81290 * yuv.b;
        color.b = yuv.r + 2.017 * yuv.g;
}

