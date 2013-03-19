#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <iostream>
#include <memory>
#include <openjpeg.h>
#include <stdexcept>

#include "video_file/j2k.h"

using namespace std;

j2k_file::j2k_file(string name) :
        m_name(name)
{
}

j2k_file::~j2k_file()
{
}

struct CharPtrDeleter
{
    void operator()(char *ptr) const
    {
        delete[] ptr;
    }
};

struct video_desc j2k_file::get_video_desc()
{
    struct video_desc desc;
    struct opj_dparameters param;

    opj_set_default_decoder_parameters(&param);

    param.cp_reduce = 0;
    param.cp_layer = 0;
    strncpy(param.infile, m_name.c_str(), OPJ_PATH_LEN);
    strncpy(param.outfile, "/dev/null", OPJ_PATH_LEN);
    /** input file format 0: J2K, 1: JP2, 2: JPT */
    param.decod_format = 0;
    /** output file format 0: PGX, 1: PxM, 2: BMP */
    param.cod_format = 2;
    /**@name JPWL decoding parameters */
    /** activates the JPWL correction capabilities */
    param.jpwl_correct = false;
    /** expected number of components */
    param.jpwl_exp_comps = 3;
    /** maximum number of tiles */
    param.jpwl_max_tiles = 1;
    /** 
      Specify whether the decoding should be done on the entire codestream, or be limited to the main header
      Limiting the decoding to the main header makes it possible to extract the characteristics of the codestream
      if == NO_LIMITATION, the entire codestream is decoded; 
      if == LIMIT_TO_MAIN_HEADER, only the main header is decoded; 
      */
    param.cp_limit_decoding = LIMIT_TO_MAIN_HEADER;

    struct stat sb;
    if(stat(m_name.c_str(), &sb) != 0) {
        throw runtime_error(sys_errlist[errno]);
    };
    std::shared_ptr<char> buffer(new char[sb.st_size], CharPtrDeleter());
    FILE *j2k_file = fopen(m_name.c_str(), "r");
    if(fread(buffer.get(), sb.st_size, 1, j2k_file) != 1) {
        throw runtime_error("Error reading file.");
    }
    fclose(j2k_file);
    opj_dinfo_t *dec = opj_create_decompress(CODEC_J2K);
    if(!dec)
        throw runtime_error("Unable to create J2K decoder!");
    opj_set_event_mgr((opj_common_ptr)dec, NULL, NULL);
    opj_setup_decoder(dec, &param);
    opj_cio_t *stream = opj_cio_open((opj_common_ptr) dec, reinterpret_cast<unsigned char *>(buffer.get()),
            sb.st_size);
    if(!stream)
        throw runtime_error("Codestream could not be opened for reading.");
    opj_codestream_info cstr_info;
    opj_image_t *image = opj_decode_with_info(dec, stream, &cstr_info);

    if(!image)
        throw runtime_error("Error decoding codestream.");

    memset(&desc, 0, sizeof(desc));
    desc.width = cstr_info.image_w;
    desc.height = cstr_info.image_h;
    desc.color_spec = J2K;
    desc.tile_count = 1;
    desc.interlacing = PROGRESSIVE;

    opj_cio_close(stream);
    opj_image_destroy(image);
    opj_destroy_decompress(dec);

    return desc;
}

char *j2k_file::get_raw_data(int &len) {
    char *ret = NULL;

    struct stat sb;
    if(stat(m_name.c_str(), &sb) != 0) {
        throw runtime_error(sys_errlist[errno]);
    };

    len = sb.st_size;

    ret = new char[len];
    int fd = open(m_name.c_str(), O_RDONLY);
    if(fd == -1) {
        throw runtime_error(sys_errlist[errno]);
    }

    ssize_t bytes_read = 0;
    do {
        ssize_t now_read = read(fd, ret + bytes_read, len - bytes_read);
        if(now_read == -1) {
            throw runtime_error(sys_errlist[errno]);
        }
        bytes_read += now_read;
    } while(bytes_read < len);
    close(fd);

    return ret;
}

