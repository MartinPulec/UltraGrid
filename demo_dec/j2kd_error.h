///
/// @file    j2kd_error.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Declaration of error-handling related stuff in JPEG 2000 decoding.
///


// prevent from multiple includes into the same file
#ifndef J2KD_ERROR_H
#define J2KD_ERROR_H

#include "j2kd_type.h"


namespace cuj2kd {



/// Represents error in process of decoding JPEG 2000 codestream.
class Error {
private:
    enum { MAX_DESCRIPTION_SIZE = 1024 * 16 };  ///< max length of description
    StatusCode status;                          ///< error code
    char message[MAX_DESCRIPTION_SIZE + 1];     ///< error message
    
public:
    /// Creates the exception, setting the description.
    Error(const StatusCode status, const char * const format, ...);
    
    /// Gets pointer to message string.
    const char * getMessage() const { return message; }
    
    /// Gets status code.
    j2kd_status_code getStatusCode() const { return status; }
}; // end of class Error



/// Throws "bad codestream" error.
void throwBadCStream(const char * fmt, ...);



/// Throws "unsupported feature required" error.
void throwUnsupported(const char * format, ...);



} // end of namespace cuj2kd

#endif // J2KD_ERROR_H

