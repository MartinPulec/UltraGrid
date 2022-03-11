This attempt splits packets to whole nal units. However, no visual advantage
was observed so it is not merged - the parsing of the buffer takes approx 0.5-3
ms, which is significant overhead (FFmpeg doesn't pass individual packets but
dumps /at least currently/ whole buffer).
