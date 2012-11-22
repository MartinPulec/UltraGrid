///
/// @file    j2kd_error.cpp
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Definition of error-handling related stuff in JPEG 2000 decoding.
///


#include <cstdarg>
#include <cstdio>
#include "j2kd_error.h"


namespace cuj2kd {



/// Creates the exception, setting the description.
Error::Error(const StatusCode status, const char * const format, ...) {
    this->status = status;
    va_list args;
    va_start(args, format);
    vsnprintf(message, MAX_DESCRIPTION_SIZE, format, args);
    va_end(args);
}



/// Throws "bad codestream" error.
void throwBadCStream(const char * format, ...) {
    enum { MAX_MESSAGE_SIZE = 16 * 1024 };
    char message[MAX_MESSAGE_SIZE + 1];
    
    // compose message
    va_list args;
    va_start(args, format);
    vsnprintf(message, MAX_MESSAGE_SIZE, format, args);
    va_end(args);
    
    // throw the exception
    throw Error(J2KD_ERROR_BAD_CODESTREAM, message);
}



/// Throws "unsupported feature required" error.
void throwUnsupported(const char * format, ...) {
    enum { MAX_MESSAGE_SIZE = 16 * 1024 };
    char message[MAX_MESSAGE_SIZE + 1];
    
    // compose message
    va_list args;
    va_start(args, format);
    vsnprintf(message, MAX_MESSAGE_SIZE, format, args);
    va_end(args);
    
    // throw the exception
    throw Error(J2KD_ERROR_UNSUPPORTED, message);
}



} // end of namespace cuj2kd

