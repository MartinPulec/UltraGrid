///
/// @file    j2kd_t2_stepsize_reader.h
/// @author  Martin Jirman  (martin.jirman@cesnet.cz)
/// @brief   Reads stepsizes from the codestream.
///

#ifndef J2KD_T2_STEPSIZE_READER_H
#define J2KD_T2_STEPSIZE_READER_H

#include "j2kd_type.h"
#include "j2kd_t2_reader.h"
#include "j2kd_error.h"

namespace cuj2kd {
    
    
    
    /// Reads stepsizes from the codestream.
    class StepsizeReader {
    private:
        /// codestream reader (for explicit and none quantization)
        T2Reader reader;
        
        // stuff for implicit quantization:
        Stepsize base;      ///< base stepsize
        int expDelta;       ///< current level's exponent delta (zero based)
        int bandsRemaining; ///< number of remaining bands in the DWT level
        
        /// Gets next stepsize for explicit quantization.
        Stepsize nextExplicit() {
            // read next two bytes and extract the stepsize from them
            if(!reader.hasBytes(2)) {
                throwBadCStream("Not enough stepsize bytes.");
            }
            const u16 code = reader.readU16();
            return Stepsize(code & 0x7FF, code >> 11);
        }
        
        /// Gets next stepsize for implicit quantization.
        Stepsize nextImplicit() {
            if(0 == --bandsRemaining) {
                bandsRemaining = 3;
                expDelta++;
            }
            return Stepsize(base.mantisa, base.exponent - expDelta);
        }
        
        /// Gets next stepsize for no quantization. (Gets exponent only.)
        Stepsize nextNone() {
            if(!reader.hasBytes(1)) {
                throwBadCStream("Not enough exponent bytes.");
            }
            return Stepsize(0, reader.readU8() >> 3);
        }
        
        /// The right method for this stepsize reader (according to type
        /// of quantization).
        Stepsize (StepsizeReader::*nextImpl)();
        
    public:
        /// Initializes stepsize reader
        StepsizeReader(const TCompQuant * const quant)
                : reader(quant->stepsizePtr, quant->stepsizeBytes) {
            // which kind of quantization?
            if(QM_NONE == quant->mode) {
                nextImpl = &StepsizeReader::nextNone;
            } else if (QM_EXPLICIT == quant->mode) {
                nextImpl = &StepsizeReader::nextExplicit;
            } else if (QM_IMPLICIT == quant->mode) {
                nextImpl = &StepsizeReader::nextImplicit;
                base = nextExplicit();
                expDelta = 0;
                bandsRemaining = 4;
            } else {
                throw Error(J2KD_ERROR_UNKNOWN, "Bad quantization mode #%d",
                            (int)quant->mode);
            }
        }
        
        /// Gets next stepsize in order of band appearance in the resolution.
        Stepsize next() { return (this->*nextImpl)(); }
    }; // end of class StepsizeReader
    
    
} // end of namespace cuj2kd

#endif // J2KD_T2_STEPSIZE_READER_H
