///
/// @file    j2kd_logger.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Logger for tracing progress of JPEG 2000 decoding.
///


#ifndef J2KD_LOGGER_H
#define J2KD_LOGGER_H

#include <cstdio>
#include <cstdarg>
#include "j2kd_timer.h"


namespace cuj2kd {


class Logger {
private:
    /// Function to be called for each composed message.
    void (*callback)(const char * message);
    
    /// maximal verbosity (inclusive) of displayed messages
    int maxVerbosity;
    
    /// Time of logger creation (in milliseconds).
    double beginTimeMs;
    
    /// Logs given message with parameters (assumes that the callback is not 
    /// null and message's verbosity is OK).
    void message(const char * type, const char * format, va_list & args) {
        enum { MAX_SIZE = 16 * 1024 };  // maximal length of the message
        char msg[MAX_SIZE + 1];         // buffer for messages's composition 
        
        // format the header of the message
        const int offset = snprintf(msg, MAX_SIZE, "CUJ2KD %s (%.2f ms): ",
                                    type, getTimeMs() - beginTimeMs);
        
        // format body of the message
        std::vsnprintf(msg + offset, MAX_SIZE - offset, format, args);
        
        // use the callback to pass the message to its destination
        callback(msg);
    }
    
    /// Default callback - prints all messages to standard output.
    static void defaultCallback(const char * message) {
        std::printf("%s\n", message);
    }
    
public:
    
    /// Default constructor with initial verbosity level.
    /// Verbosity levels:
    ///   0 = quiet
    ///   1 = decoder error
    ///   2 = decoder warning
    ///   3 = decoder timing
    ///   4 = decoder info
    ///   5 = decoder debug
    /// @param maxVerbosity maximal verbosity of displayed messages (inclusive)
    Logger(const int maxVerbosity = 2) {
        this->callback = defaultCallback;
        this->maxVerbosity = maxVerbosity;
        this->beginTimeMs = getTimeMs();
    }
    
    /// Logs formatted error message.
    /// @param format format string (same as for printf).
    void error(const char * const format, ...) {
        if(callback && maxVerbosity >= 1) {
            va_list args;
            va_start (args, format);
            message("ERROR", format, args);
            va_end (args);
        }
    }
    
    /// Logs formatted warning.
    /// @param format format string (same as for printf).
    void warning(const char * const format, ...) {
        if(callback && maxVerbosity >= 2) {
            va_list args;
            va_start (args, format);
            message("WARNING", format, args);
            va_end (args);
        }
    }
    
    /// Logs formatted timing info.
    /// @param format format string (same as for printf).
    void time(const char * const format, ...) {
        if(callback && maxVerbosity >= 3) {
            va_list args;
            va_start (args, format);
            message("TIME", format, args);
            va_end (args);
        }
    }
    
    /// Logs formatted info message.
    /// @param format format string (same as for printf).
    void info(const char * const format, ...) {
        if(callback && maxVerbosity >= 4) {
            va_list args;
            va_start (args, format);
            message("INFO", format, args);
            va_end (args);
        }
    }
    
    /// Logs formatted debug message.
    /// @param format format string (same as for printf).
    void debug(const char * const format, ...) {
        if(callback && maxVerbosity >= 5) {
            va_list args;
            va_start (args, format);
            message("DEBUG", format, args);
            va_end (args);
        }
    }
    
    /// Sets new log message callback.
    void setCallback(void (*newCallback)(const char * message)) {
        callback = newCallback;
    }
    
    /// Sets new logger verbosity.
    void setVerbosity(const int newVerbosity) {
        maxVerbosity = newVerbosity;
    }
}; // end of class Logger
    

} // end of namespace cuj2kd


#endif // J2KD_LOGGER_H

