Third party sources and external libraries
==========================================

UltraGrid
--------

### MD5

Files:
- src/crypto/md5.c
- src/crypto/md5.h

Copyright:
- 1991-2, RSA Data Security, Inc.

Created 1991

License:
````
License to copy and use this software is granted provided that it is
identified as the "RSA Data Security, Inc. MD5 Message-Digest Algorithm"
in all material mentioning or referencing this software or this function.

License is also granted to make and use derivative works provided that such
works are identified as "derived from the RSA Data Security, Inc. MD5
Message-Digest Algorithm" in all material mentioning or referencing the
derived work.

RSA Data Security, Inc. makes no representations concerning either the
merchantability of this software or the suitability of this software for
any particular purpose. It is provided "as is" without express or implied
warranty of any kind.

These notices must be retained in any copies of any part of this
documentation and/or software.
````

## Random

Files:
- src/crypto/random.c

- src/crypto/random.h
Copyright:
- 1993 Regents of the University of California
- Computer Systems Engineering Group at Lawrence Berkeley Laboratory

License: BSD-4-Clause

### DXT

Files:
- dxt_compress/*glsl except yuv422_to_yuv444.glsl

Copyright:
-  NVIDIA Corporation

Real-time DXT1 & YCoCg-DXT5 compression (Cg 2.0)

Written by: Ignacio Castano 

Thanks to JMP van Waveren, Simon Green, Eric Werness, Simon Brown

License: MIT

Files:
- dxt_compress/* except the above

Copyright:
- 2011, Martin Srom

License: BSD-2-Clause

External libraries
------------------

### EmbeddableWebServer

Copyright:
- 2016, 2019, 2020 Forrest Heller, and CONTRIBUTORS

License: BSD-2-Clause

Contributors:
- Martin Pulec - bug fixes, warning fixes, IPv6 support, msys2/mingw support
- Daniel Barry - bug fix (ifa_addr != NULL)

### FFmpeg

(taken from debian/copyright ffmpeg package version 8.1.2-2 as is)

```
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: FFmpeg
Upstream-Contact: ffmpeg-devel@ffmpeg.org
Source: https://ffmpeg.org
Comment:
 Most files in FFmpeg are licensed under the GNU Lesser General Public License
 version 2.1 or later (LGPL v2.1+), but some optional parts of FFmpeg are
 licensed under the GNU General Public License version 2 or later (GPL v2+)
 and a few have different, but compatible licenses.
 .
 For building the default Debian packages some of the GPL licensed files are used,
 so the resulting binaries are licensed under GPL v2+.
 For libavcodec/libavfilter an extra flavor is built, which links against external
 libraries licensed under the Apache License 2.0, which makes it effectively licensed
 under the GPL v3+
 The HTML documentation /usr/share/doc/ffmpeg/manual/*.html in the ffmpeg-doc
 package is effectively licensed under GPL-3+, because during its creation
 the GPL-3+ licensed file doc/t2h.pm is used.
 .
 FFmpeg can also be combined with non-free libraries, which would make the
 resulting binaries unredistributable.
 But this is not done for the Debian packages.

Files: *
Copyright:
 1990, James Ashton - Sydney University
 1991-1999, Free Software Foundation, Inc
 1993, Computer Science, Speech Group
 1994-2012, the Xiph.Org Foundation and contributors
 1996-2002, Gerd Knorr
 1997-1999, H. Dietz
 1997-1999, R. Fisher
 1998-2009, Conifer Software
 1999, Nick Bailey
 1999, Roger Hardiman
 1999-2000, Sebastien Rougeaux <sebastien.rougeaux@anu.edu.au>
 1999-2001, Chris Bagwell <cbagwell@sprynet.com>
 1999-2024, Intel Corporation
 2000, Chris Ausbrooks <weed@bucket.pp.ualr.edu>
 2000, Fabien COELHO <fabien@coelho.net>
 2000, John Walker
 2000-2001, Michel Lespinasse <walken@zoy.org>
 2000-2001, Peter Gubanov <peter@elecard.net.ru>
 2000-2003, Nick Kurshev <nickols_k@mail.ru>
 2000-2010, Fabrice Bellard <fabrice@bellard.org>
 2001, Dan Dennedy <dan@dennedy.org>
 2001, Heikki Leinonen
 2001, Juan J. Sierralta P
 2001, Lionel Ulmer
 2001, Michel Lespinasse
 2001, Tim Ferguson
 2001-2003, BERO <bero@geocities.co.jp>
 2001-2006, Daniel Maas <dmaas@dcine.com>
 2001-2010, Vladimir Sadovnikov
 2001-2024, Michael Niedermayer <michaelni@gmx.at>
 2001-2025, Peter Ross <pross@xvid.org>
 2001-2025, The FFmpeg project
 2002, Anders Johansson <ajh@atri.curtin.edu.au>
 2002, Brian Foley
 2002, Daniel Pouzzner
 2002, Dieter Shirley
 2002, Falk Hueffner <falk@debian.org>
 2002, Frederic 'dilb' Boulay <dilb@handhelds.org>
 2002, Gunnar Monell <gmo@linux.nu>
 2002, Laszlo Torok <torokl@alpha.dfmk.hu>
 2002, Lennert Buytenhek <buytenh@gnu.org>
 2002, Mark Hills <mark@pogo.org.uk>
 2002, Naoki Shibata
 2002, Remi Guyomarch <rguyom@pobox.com>
 2002, Steve O'Hara-Smith
 2002, The Xine project
 2002, Zdenek Kabelac <kabi@informatics.muni.cz>
 2002-2005, Francois Revol <revol@free.fr>
 2002-2005, Roberto Togni
 2002-2006, Alex Beregszaszi
 2002-2013, Maxim Poliakovski
 2003, Alex Beregszaszi
 2003, Donnie Smith
 2003, Dr. Tim Ferguson
 2003, Ewald Snel
 2003, Gustavo Sverzut Barbieri <gsbarbieri@yahoo.com.br>
 2003, International Business Machines, Corp
 2003, James Klicman <james@klicman.org>
 2003, Max Krasnyansky <maxk@qualcomm.com>
 2003, Mike Melanson
 2003, Nick Kurshev
 2003, Rich Felker
 2003, Thomas Raivio
 2003, Tinic Uro
 2003-2004, Romain Dolbeau <romain@dolbeau.org>
 2003-2005, Christopher R. Hertel <crh@ubiqx.mn.org>
 2003-2006, Roman Shaposhnik
 2003-2007, Michel Bardiaux <mbardiaux@mediaxim.be>
 2003-2007, Mike Melanson <melanson@pcisys.net>
 2003-2011, Sascha Sommer <saschasommer@freenet.de>
 2003-2014, Loren Merritt <lorenm@u.washington.edu>
 2003-2014, Pascal Massimino <skal@planet-d.net>
 2003-2017, Ivan Kalvachev
 2003-2024, x264 project
 2004, AGAWA Koji <i@atty.jp>
 2004, Adam Thayer <krevnik@comcast.net>
 2004, Benjamin Zores
 2004, Gildas Bazin <gbazin@videolan.org>
 2004, Maarten Daniels
 2004-2005, Denes Balatoni <dbalatoni@programozo.hu>
 2004-2005, Henryk Ploetz <henryk@ploetzli.ch>
 2004-2006, Lennart Poettering
 2004-2007, Benjamin Zores <ben@geexbox.org>
 2004-2007, Eric Lasota
 2004-2007, Marc Hoffman <marc.hoffman@analog.com>
 2004-2010, Marcel Holtmann <marcel@holtmann.org>
 2004-2014, Konstantin Shishkov <kostya.shishkov@gmail.com>
 2005, Alban Bedel <albeu@free.fr>
 2005, Balatoni Denes
 2005, Boðaç Topaktaþ
 2005, David Hammerton
 2005, DivX, Inc
 2005, Jeff Muizelaar
 2005, Matthieu CASTET
 2005, Neal Symms <tivo@freakinzoo.com>
 2005, Ole André Vadla Ravnås <oleavr@gmail.com>
 2005, Roine Gustafsson
 2005, Steve Underwood <steveu@coppice.org>
 2005, Vidar Madsen
 2005, Wim Taymans
 2005, Zoltan Hidvegi <hzoli@hzoli.com>
 2005-2006, Oded Shimon <ods15@ods15.dyndns.org>
 2005-2006, Robert Edele <yartrebo@earthlink.net>
 2005-2007, Wolfram Gloger
 2005-2008, Brad Midgley <bmidgley@xmission.com>
 2005-2008, Ian Caulfield
 2005-2011, Benjamin Larsson
 2005-2012, Laurent de Soras
 2005-2012, Måns Rullgård <mans@mansr.com>
 2005-2012, VLC authors and VideoLAN
 2005-2013, Diego Biurrun <diego@biurrun.de>
 2005-2015, Luca Barbato <lu_zero@gentoo.org>
 2005-2020, Reimar Döffinger <Reimar.Doeffinger@gmx.de>
 2006, Benjamin Larssonb
 2006, Corey Hickey
 2006, Cyril Zorin
 2006, Guillaume Poirier <gpoirier@mplayerhq.hu>
 2006, Industrial Light & Magic, a division of Lucas Digital Ltd. LLC
 2006, John Maddock
 2006, Kartikey Mahendra BHATT <bhattkm@gmail.com>
 2006, Patrick Guimond
 2006, Paul Richards <paul.richards@gmail.com>
 2006, Thijs Vermeir <thijs.vermeir@barco.com>
 2006-2007, Maxim Gavrilov <maxim.gavrilov@gmail.com>
 2006-2007, Reynaldo H. Verdejo Pinochet
 2006-2007, Ryan Martell <rdm4@martellventures.com>
 2006-2008, Gregory Montoir <cyx@users.sourceforge.net>
 2006-2009, Luca Abeni <lucabe72@email.it>
 2006-2009, Robert Swain <rob@opendot.cl>
 2006-2009, Stefan Gehrer <stefan.gehrer@gmx.de>
 2006-2010, Prakash Punnoor <prakash@punnoor.de>
 2006-2011, Baptiste Coudurier <baptiste.coudurier@free.fr>
 2006-2011, Xvid Solutions GmbH
 2006-2012, Rob Sykes <robs@users.sourceforge.net>
 2006-2013, Justin Ruggles <justin.ruggles@gmail.com>
 2006-2014, Ramiro Polla <ramiro.polla@gmail.com>
 2006-2015, Steve Lhomme
 2006-2017, Aurelien Jacobs <aurel@gnuage.org>
 2007, Alexis Ballier
 2007, Christian Ohm
 2007, Clemens Fruhwirth <clemens@endorphin.org>
 2007, Kamil Nowosad
 2007, Måns Rullgård
 2007, Nicholas Tung
 2007, Philippe Kalaf
 2007, Reynaldo H. Verdejo Pinochet (QCELP decoder)
 2007, Richard Spindler
 2007, Ulion
 2007-2008, CSIRO
 2007-2008, Ivo van Poorten
 2007-2008, Joseph Artsimovich <jart@users.sourceforge.net>
 2007-2008, Marco Gerards <marco@gnu.org>
 2007-2008, Siarhei Siamashka <ssvb@users.sourceforge.net>
 2007-2008, UAB "DKD
 2007-2009, Bartlomiej Wolowiec <bartek.wolowiec@gmail.com>
 2007-2009, Björn Axelsson
 2007-2009, Xiph.Org Foundation
 2007-2010, Bobby Bingham
 2007-2010, David Conrad <lessen42@gmail.com>
 2007-2010, Nokia Corporation
 2007-2011, Vitor Sessak <vitor1001@gmail.com>
 2007-2011, Vladimir Voroshilov
 2007-2013, Stefano Sabatini <stefano.sabatini-lala@poste.it>
 2007-2014, Benoit Fouet <benoit.fouet@free.fr>
 2007-2015, Christophe Gisquet
 2007-2015, Collabora Ltd
 2007-2015, Edward Hervey
 2007-2016, David Robillard <http://drobilla.net>
 2007-2018, Ronald S. Bultje <rsbultje@gmail.com>
 2007-2020, Anssi Hannula <anssi.hannula@iki.fi>
 2007-2020, Nicolas George <nicolas.george@normalesup.org>
 2008, Affine Systems, Inc (Michael Sullivan, Bobby Impollonia)
 2008, Alessandro Sappia
 2008, BBC, Anuradha Suraparaju <asuraparaju@gmail.com>
 2008, Jaikrishnan Menon
 2008, NVIDIA
 2008, Rob Sykes
 2008, Robert Marston
 2008, Sisir Koppaka
 2008, Victor Paesa
 2008, vmrsss
 2008-2009, Andrej Stepanchuk
 2008-2009, Gregory Maxwell
 2008-2009, Jaikrishnan Menon <realityman@gmx.net>
 2008-2009, Splitted-Desktop Systems
 2008-2010, Alexander Strange <astrange@ithinksw.com>
 2008-2010, Eli Friedman <eli.friedman@gmail.com>
 2008-2010, Paul Kendall <paul@kcbbs.gen.nz>
 2008-2011, Zhentan Feng <spyfeng@gmail.com>
 2008-2012, Alexander E. Patrakov
 2008-2012, Laurent Aimar <fenrir@videolan.org>
 2008-2013, Alex Converse <alex.converse@gmail.com>
 2009, Alex Converse
 2009, Colin McQuillan
 2009, Colin McQuillian
 2009, Dylan Yudaken
 2009, Giliard B. de Freitas <giliarde@gmail.com>
 2009, Jimmy Christensen
 2009, Joshua Warner
 2009, Kenan Gillet
 2009, Michael Tison
 2009, Naotoshi Nojiri
 2009, Nicolas Martin <martinic@iro.umontreal.ca>
 2009, Peter Holik
 2009, Samalyse
 2009, Sebastien Lucas <sebastien.lucas@gmail.com>
 2009, Stephen Backway
 2009, Thomas P. Higdon <thomas.p.higdon@gmail.com>
 2009, Tobias Bindhammer
 2009, Toshimitsu Kimura
 2009, Xuggle Incorporated
 2009, Zuxy Meng <zuxy.meng@gmail.com>
 2009-2010, Fiona Glaser <fiona@x264.com>
 2009-2010, Howard Chu
 2009-2011, Sebastian Gesemann
 2009-2012, Nathan Caldwell <saintdev@gmail.com>
 2009-2013, Christian Schmidt
 2009-2013, Daniel Verkamp <daniel@drv.nu>
 2009-2015, James Darnley <james.darnley@gmail.com>
 2009-2018, Kostya Shishkov
 2009-2023, Thilo Borgmann <thilo.borgmann@mail.de>
 2009-2024, Martin Storsjo <martin@martin.st>
 2010, Adrian Daerr
 2010, Amanda, Y.N. Wu <amanda11192003@gmail.com>
 2010, Andrzej Szombierski
 2010, Brandon Mintern
 2010, Daniel G. Taylor <dan@programmer-art.org>
 2010, Francesco Lavra <francescolavra@interfree.it>
 2010, Hans de Goede <hdegoede@redhat.com>
 2010, Holger Lubitz
 2010, Jacob Meuser
 2010, Josh Allmann
 2010, Kenneth Vermeirsch
 2010, Marcelo Galvao Povoa
 2010, Mark Heath <mjpeg0@silicontrip.org>
 2010, Mark Nauwelaerts
 2010, Michael Chinen
 2010, Michele Orrù
 2010, Mohamed Naufal Basheer <naufal11@gmail.com>
 2010, Nolan Lum <nol888@gmail.com>
 2010, Rob Clark <rob@ti.com>
 2010, Romain Degez
 2010, S.N. Hemanth Meenakshisundaram <smeenaks@ucsd.edu>
 2010, Sebastian Vater <cdgs.basty@googlemail.com>
 2010-2011, Anatoly Nenashev
 2010-2011, Elvis Presley
 2010-2012, Rafaël Carré <funman@videolan.org>
 2010-2013, Georg Martius <georg.martius@web.de>
 2010-2015, Janne Grunau <janne-libav@jannau.net>
 2010-2017, Carl Eugen Hoyos
 2010-2019, Philip Langdale <philipl@overt.org>
 2010-2020, Google, Inc
 2010-2023, Anton Khirnov
 2010-2023, Tomas Härdin
 2010-2024, Anton Khirnov <anton@khirnov.net>
 2010-2024, Rémi Denis-Courmont
 2011, Anatoliy Wasserman
 2011, Andreas Öman
 2011, David Goldwich
 2011, Janne Grunau
 2011, Jordi Ortiz
 2011, Juan Carlos Rodriguez <ing.juancarlosrodriguez@hotmail.com>
 2011, Matthew Hoops <clone2727@gmail.com>
 2011, Max Horn
 2011, Michael Karcher
 2011, Mina Nagy Zaki
 2011, Miroslav Slugeň <Thunder.m@seznam.cz>
 2011, MirriAd Ltd
 2011, Oskar Arvidsson
 2011, Roger Pau Monné <roger.pau@entel.upc.edu>
 2011, Sven Hesse <drmccoy@drmccoy.de>
 2011, Thomas Kuehnel
 2011, Xiang Wang
 2011-2012, Hyllian/Jararaca <sergiogdb@gmail.com>
 2011-2012, Mark Himsley
 2011-2012, Mashiat Sarker Shakkhar
 2011-2012, Michael Bradshaw <mjbshaw@gmail.com>
 2011-2012, Smartjog S.A.S, Clément Bœsch <clement.boesch@smartjog.com>
 2011-2013, Daniel Kang <daniel.d.kang@gmail.com>
 2011-2016, Kieran Kunhya <kieran@kunhya.com>
 2011-2017, KO Myung-Hun <komh@chollian.net>
 2011-2019, Derek Buitenhuis
 2011-2022, Clément Bœsch <u@pkh.me>
 2011-2025, Paul B Mahol
 2012, Aleksi Nurmi
 2012, Andrew D'Addesio
 2012, Antti Seppälä
 2012, AvxSynth Team
 2012, British Broadcasting Corporation
 2012, David Kment
 2012, Jeremy Tran
 2012, Krzysztof Klinikowski
 2012, Nathan Caldwell
 2012, Pavel Koshevoy <pkoshevoy@gmail.com>
 2012, Petri Hintukainen <phintuka users.sourceforge.net>
 2012, Robert Nagy <ronag89@gmail.com>
 2012, Rudolf Polzer
 2012, Samuel Pitoiset
 2012, Sebastien Zwickert
 2012, Steven Robertson
 2012, Vitaliy E Sugrobov
 2012, Youness Alaoui <kakaroto@kakaroto.homelinux.net>
 2012-2013, Andrey Utkin <andrey.krieger.utkin@gmail.com>
 2012-2013, Aneesh Dogra (lionaneesh) <lionaneesh@gmail.com>
 2012-2013, Fredrik Mellbin
 2012-2013, Gildas Cocherel
 2012-2013, Guillaume Martres <smarter@ubuntu.com>
 2012-2013, Mickael Raulet
 2012-2013, Oka Motofumi <chikuzen.mo@gmail.com>
 2012-2013, Rudolf Polzer <divverent@xonotic.org>
 2012-2013, Wassim Hamidouche
 2012-2014, Georg Lippitsch <georg.lippitsch@gmx.at>
 2012-2014, Petri Hintukainen <phintuka@users.sourceforge.net>
 2012-2016, Ben "GreaseMonkey" Russell
 2012-2023, Jan Ekström
 2012-2024, James Almer <jamrial@gmail.com>
 2013, Anand Meher Kotra
 2013, Aneesh Dogra <aneesh@sugarlabs.org>
 2013, Ash Hughes
 2013, Darryl Wallace <wallacdj@gmail.com>
 2013, Dirk Farin <dirk.farin@gmail.com>
 2013, Jeff Moguillansky
 2013, Lenny Wang <lwanghpc@gmail.com>
 2013, MIPS Technologies, Inc., California
 2013, Matthew Heaney
 2013, Rl, Aetey Global Technologies AB
 2013, VTT
 2013, Vadim Kalinsky <vadim@kalinsky.ru>
 2013, Wei Gao <weigao@multicorewareinc.com>
 2013, Xiaolei Yu <dreifachstein@gmail.com>
 2013, Yukinori Yamazoe
 2013-2014, Andrew Kelley
 2013-2014, Deti Fliegl
 2013-2014, Lukasz Marek <lukasz.m.luki@gmail.com>
 2013-2014, Mozilla Corporation
 2013-2014, Nicolas Bertrand <nicoinattendu@gmail.com>
 2013-2014, Pierre-Edouard Lepere
 2013-2014, RISC OS Open Ltd
 2013-2015, Andreas Fuchs
 2013-2015, Seppo Tomperi <seppo.tomperi@vtt.fi>
 2013-2015, Tiancheng "Timothy" Gu <timothygu99@gmail.com>
 2013-2015, Wolfgang Hrauda
 2013-2017, Vittorio Giovara <vittorio.giovara@gmail.com>
 2013-2018, Calvin Walton <calvin.walton@kepstin.ca>
 2013-2020, Marton Balint
 2013-2020, Michael Barbour <barbour.michael.0@gmail.com>
 2013-2022, Andreas Unterweger
 2013-2022, Ben Avison <bavison@riscosopen.org>
 2014, Aman Gupta <ffmpeg@tmm1.net>
 2014, Daniel Oberhoff
 2014, Dave Rice
 2014, Eejya Singh
 2014, James Yu <james.yu@linaro.org>
 2014, Marvin Scholz
 2014, Nicholas Robbins
 2014, Nicolas Bertrand
 2014, Oleksij Rempel <linux@rempel-privat.de>
 2014, Rong Yan
 2014, Samsung Electronics
 2014, StarBrilliant <m13253@hotmail.com>
 2014, Tim Walker <tdskywalker@gmail.com>
 2014-2015, Arwa Arif <arwaarif1994@gmail.com>
 2014-2015, Peter Meerwald <pmeerw@pmeerw.net>
 2014-2015, Supraja Meedinti
 2014-2015, Vignesh Venkatasubramanian
 2014-2016, Muhammad Faiz <mfcc64@gmail.com>
 2014-2016, Neil Birkbeck <birkbeck@google.com>
 2014-2017, Alexandra Hájková
 2014-2018, Thomas Volkert <thomas@homer-conferencing.com>
 2014-2020, Hendrik Leppkes
 2014-2021, Jason Jang
 2015, Andreas Cadhalpun <Andreas.Cadhalpun@googlemail.com>
 2015, Anshul Maheshwari
 2015, Claudio Freire
 2015, Donny Yang
 2015, Eran Kornblau <erankor@gmail.com>
 2015, Gilles Chanteperdrix <gch@xenomai.org>
 2015, Himangi Saraogi <himangi774@gmail.com>
 2015, Imagination Technologies Ltd
 2015, Kevin Wheatley <kevin.j.wheatley@gmail.com>
 2015, LoRd_MuldeR <mulder2@gmx.de>
 2015, Mats Peterson
 2015, Matthew Waters <matthew@centricular.com>
 2015, Parag Salasakar <parag.salasakar@imgtec.com>
 2015, Pedro Arthur <bygrandao@gmail.com>
 2015, Rick Kern <kernrj@gmail.com>
 2015, Roger Pack
 2015, Rostislav Pehlivanov
 2015, Sebastian Dröge <sebastian@centricular.com>
 2015, Stupeflix
 2015, Tampere University of Technology
 2015, Tom Butterworth <bangnoise@gmail.com>
 2015, Urvang Joshi
 2015, Vesselin Bontchev
 2015, Zhang Rui <bbcallen@gmail.com>
 2015, Zhang Shuangshuang <zhangshuangshuang@ict.ac.cn>
 2015-2016, Florian Nouwt
 2015-2016, Ganesh Ajjanagadde <gajjanaq@gmail.com>
 2015-2016, Kyle Swanson <k@ylo.ph>
 2015-2016, Open Broadcast Systems Ltd
 2015-2016, Timo Rothenpieler <timo@rothenpieler.org>
 2015-2016, Zhou Xiaoyong <zhouxiaoyong@loongson.cn>
 2015-2017, Manojkumar Bhosale <Manojkumar.Bhosale@imgtec.com>
 2015-2017, Matthieu Bouron <matthieu.bouron stupeflix.com>
 2015-2017, Parag Salasakar <Parag.Salasakar@imgtec.com>
 2015-2017, Shivraj Patil <Shivraj.Patil@imgtec.com>
 2015-2018, Henrik Gramner <henrik@gramner.com>
 2015-2018, Rostislav Pehlivanov <atomnuker@gmail.com>
 2015-2021, Facebook, Inc
 2015-2021, Rodger Combs <rodger.combs@gmail.com>
 2015-2021, rcombs
 2015-2024, Loongson Technology Corporation Limited
 2016, Davinder Singh (DSM_) <ds.mudhar<@gmail.com>
 2016, Floris Sluiter
 2016, Jan Sebechlebsky
 2016, Josh de Kock
 2016, KongQun Yang <kqyang@google.com>
 2016, Marton Balnt <cus@passwd.hu>
 2016, Mobibase, France
 2016, Neil Birkbeck <neil.birkbeck@gmail.com>
 2016, Ståle Kristoffersen
 2016, Thomas Mundt <loudmax@yahoo.de>
 2016, Thomas Volkert <thomas@netzeal.de>
 2016, Tobias Rapp
 2016, Umair Khan <omerjerk@gmail.com>
 2016, William Ma, Sofia Kim, Dustin Woo
 2016, William Ma, Ted Ying, Jerry Jiang
 2016-2017, Savoir-faire Linux, Inc
 2016-2017, foo86
 2016-2018, Jokyo Images
 2016-2019, Jai Luthra
 2017, <samsamsam@o2.pl>
 2017, Adib Surani
 2017, Alexis Ballier <aballier@gentoo.org>
 2017, Ashish Pratap Singh <ashk43712@gmail.com>
 2017, Daniil Cherednik
 2017, Felix Matouschek
 2017, Ivan Kalvachev <ikalvachev@gmail.com>
 2017, Jorge Ramirez <jorge.ramirez-ortiz@linaro.org>
 2017, Kaustubh Raste <kaustubh.raste@imgtec.com>
 2017, Lionel CHAZALLON
 2017, Maksym Veremeyenko
 2017, Meng Wang <wangmeng.kids@bytedance.com>
 2017, Paras Chadha
 2017, Richard Ling
 2017, Steinar H. Gunderson
 2017, Steven Liu
 2017, sfan5 <sfan5@live.de>
 2017-2018, Akamai Technologies, Inc
 2018, Bjorn Roche
 2018, Danil Iashchenko
 2018, Dylan Fernando
 2018, Falei Luo <falei.luo@gmail.com>
 2018, Magnus Röös <mla2.roos@gmail.com>
 2018, Mina Sami
 2018, Misty De Meo
 2018, Mohammad Izadi <moh.izadi@gmail.com>
 2018, Sergey Lavrushkin
 2018, Yingming Fan <yingmingfan@gmail.com>
 2018, Yiqun Xu <yiqun.xu@vipl.ict.ac.cn>
 2018-2019, Shiyou Yin <yinshiyou-hf@loongson.cn>
 2018-2019, gxw <guxiwei-hf@loongson.cn>
 2018-2020, Huiwen Ren <hwrenx@gmail.com>
 2018-2025, softworkz
 2019, Eugene Lyapustin
 2019, Guo Yejun
 2019, Leo Zhang <leozhang@qiyi.com>
 2019, Swaraj Hota
 2019, Vladimir Panteleev
 2019-2020, Andriy Gelman
 2019-2021, Sebastian Pop <spop@amazon.com>
 2019-2021, Xuewei Meng
 2019-2022, Manoj Gupta Bonda
 2019-2025, Lynne <dev@lynne.ee>
 2020, Alyssa Milburn <amilburn@zall.org>
 2020, Aman Karmani <aman@tmm1.net>
 2020, Björn Ottosson
 2020, David Bryant
 2020, Gautam Ramakrishnan
 2020, John Stebbins <jstebbins.hb@gmail.com>
 2020, Jun Zhao <barryjzhao@tencent.com>
 2020, Nelson Gomez <nelson.gomez@microsoft.com>
 2020, Stefan Dyulgerov <stefan.dyulgerov@gmail.com>
 2020, Vacing Fang <vacingfang@tencent.com>
 2020, Yaroslav Pogrebnyak <yyyaroslav@gmail.com>
 2020, Zhenyu Wang <wangzhenyu@pkusz.edu.cn>
 2020, Zixing Liu
 2020-2021, 24i
 2020-2021, Zane van Iperen <zane@zanevaniperen.com>
 2020-2022, Andreas Rheinhardt
 2020-2024, J. Dekker <jdek@itanimul.li>
 2020-2024, Nuo Mi <nuomi2021@gmail.com>
 2021, Aidan Richmond
 2021, Boris Baracaldo
 2021, Dawid Kozinski
 2021, Limin Wang <lance.lmwang@gmail.com>
 2021, Mark Reid <mindmark@gmail.com>
 2021, Nachiket Tarate
 2021, Paul Buxton
 2021, Pekka Väänänen <pekka.vaananen@iki.fi>
 2021, VideoLAN and dav1d authors
 2021, parazyd <parazyd@dyne.org>
 2021, quietvoid
 2021-2023, Nuomi
 2021-2024, Leo Izen <leo.izen@gmail.com>
 2021-2024, Wu Jianhua <jianhua.wu@intel.com>
 2021-2025, Dawid Kozinski <d.kozinski@samsung.com>
 2021-2025, Niklas Haas <ffmpeg@haasn.xyz>
 2021-2025, Two Orioles, LLC
 2022, Caleb Etemesi <etemesicaleb@gmail.com>
 2022, Erik Linge
 2022, Jack Bruienne
 2022, Jonathan Swinney <jswinney@amazon.com>
 2022, Mark Gaiser
 2022, Mohamed Khaled <Mohamed_Khaled_Kamal@outlook.com>
 2022, Pierre-Anthony Lemieux <pal@palemieux.com>
 2022, TADANO Tokumei
 2022, Thomas Siedel
 2022, Victoria Zhislina, Intel
 2022, Xu Mu
 2022-2025, Zhao Zhili <quinkblack@foxmail.com>
 2023, Elias Carotti <eliascrt@amazon.it>
 2023, Francesco Carusi
 2023, Jan Ekström <jeebjp@gmail.com>
 2023, John Cox <jc@kynesim.co.uk>
 2023, LTN Global Communications
 2023, Loongson Technology Co. Ltd
 2023, SiFive, Inc
 2023, xu fulong
 2023-2024, Institute of Software Chinese Academy of Sciences (ISCAS)
 2023-2024, Leo Izen
 2024, Antoine Soulier <asoulier@google.com>
 2024, Axis Communications
 2024, Christian Lehmann
 2024, Christian R. Helmrich
 2024, Christian Stoffers
 2024, Connor Worley
 2024, Martin Schitter
 2024, Ruslan Chernenko
 2024, Shaun Loo
 2024, Stone Chen
 2024, asivery
 2024-2025, Emma Worley <emma@emma.gg>
 2025, Jack Lau
 2025, Jacob Lifshay
 2025, Jonathan Baudanza <jon@jonb.org>
 2025, Krzysztof Aleksander Pyrkosz
 2025, MulticorewWare, Inc
 2025, Vittorio Palmisano
 Acoustics Research Institute (ARI), Vienna, Austria
 Alexandra Hajkova
 Edward Hervey <bilboed@gmail.com>
 Matthieu Bouron <matthieu.bouron@gcollabora.com>
 Reynaldo H. Verdejo Pinochet <r.verdejo@sisa.samsung.com>
 Bingjie Han <hanbj@pkusz.edu.cn>
 Chengxiang Lu and Alex Hauptmann
 Christian Holschuh
 Edouard Auvinet
 Heiher <r@hev.cc>
 Joseph Artsimovich and UAB "DKD"
 Markus Schmidt and Christian Holschuh
 Mohamed Naufal <naufal22@gmail.com>
 Sebastien Bechet <s.bechet@av7.net>
 Vitor Sessak <vitor1001 gmail com>
License: LGPL-2.1+
Comment:
 The copyright details for Joseph Artsimovich and UAB "DKD" were extracted
 from the referenced source:
 <http://samples.mplayerhq.hu/A-codecs/Nelly_Moser/ASAO/ASAO.zip>.

Files:
 libavfilter/af_chorus.c
 libavfilter/af_earwax.c
Copyright:
 1998, Juergen Mueller And Sundry Contributors
 2000, Edward Beingessner And Sundry Contributors
 2011, Mina Nagy Zaki
 2015, Paul B Mahol
License: LGPL-2.1+ and Sundry

Files:
 doc/texi2pod.pl
 libavcodec/aarch64/ac3dsp_init_aarch64.c
 libavcodec/aarch64/ac3dsp_neon.S
 libavcodec/x86/flac_dsp_gpl.asm
 libavfilter/f_ebur128.c
 libavfilter/perlin.c
 libavfilter/phase_template.c
 libavfilter/qp_table.c
 libavfilter/qp_table.h
 libavfilter/signature.h
 libavfilter/signature_lookup.c
 libavfilter/tinterlace.h
 libavfilter/vf_blackframe.c
 libavfilter/vf_boxblur.c
 libavfilter/vf_colormatrix.c
 libavfilter/vf_cover_rect.c
 libavfilter/vf_cropdetect.c
 libavfilter/vf_delogo.c
 libavfilter/vf_eq.c
 libavfilter/vf_eq.h
 libavfilter/vf_find_rect.c
 libavfilter/vf_fspp.c
 libavfilter/vf_fsppdsp.c
 libavfilter/vf_fsppdsp.h
 libavfilter/vf_histeq.c
 libavfilter/vf_hqdn3d.c
 libavfilter/vf_hqdn3d.h
 libavfilter/vf_kerndeint.c
 libavfilter/vf_mcdeint.c
 libavfilter/vf_mpdecimate.c
 libavfilter/vf_nnedi.c
 libavfilter/vf_owdenoise.c
 libavfilter/vf_perspective.c
 libavfilter/vf_phase.c
 libavfilter/vf_pp7.c
 libavfilter/vf_pp7.h
 libavfilter/vf_pullup.c
 libavfilter/vf_pullup.h
 libavfilter/vf_repeatfields.c
 libavfilter/vf_sab.c
 libavfilter/vf_signature.c
 libavfilter/vf_smartblur.c
 libavfilter/vf_spp.c
 libavfilter/vf_spp.h
 libavfilter/vf_stereo3d.c
 libavfilter/vf_super2xsai.c
 libavfilter/vf_tinterlace.c
 libavfilter/vf_uspp.c
 libavfilter/vf_vaguedenoiser.c
 libavfilter/vsrc_mptestsrc.c
 libavfilter/x86/vf_eq.asm
 libavfilter/x86/vf_eq_init.c
 libavfilter/x86/vf_fspp.asm
 libavfilter/x86/vf_fspp_init.c
 libavfilter/x86/vf_hqdn3d_init.c
 libavfilter/x86/vf_interlace.asm
 libavfilter/x86/vf_pp7.asm
 libavfilter/x86/vf_pp7_init.c
 libavfilter/x86/vf_pullup.asm
 libavfilter/x86/vf_pullup_init.c
 libavfilter/x86/vf_removegrain.asm
 libavfilter/x86/vf_spp.c
 libavfilter/x86/vf_tinterlace_init.c
 libavutil/macos_kperf.c
 libavutil/macos_kperf.h
 libswresample/tests/*
 tests/checkasm/aacencdsp.c
 tests/checkasm/aacpsdsp.c
 tests/checkasm/aarch64/*
 tests/checkasm/ac3dsp.c
 tests/checkasm/aes.c
 tests/checkasm/af_afir.c
 tests/checkasm/alacdsp.c
 tests/checkasm/arm/*
 tests/checkasm/audiodsp.c
 tests/checkasm/av_tx.c
 tests/checkasm/blockdsp.c
 tests/checkasm/bswapdsp.c
 tests/checkasm/checkasm.c
 tests/checkasm/checkasm.h
 tests/checkasm/diracdsp.c
 tests/checkasm/exrdsp.c
 tests/checkasm/fdctdsp.c
 tests/checkasm/fixed_dsp.c
 tests/checkasm/flacdsp.c
 tests/checkasm/float_dsp.c
 tests/checkasm/fmtconvert.c
 tests/checkasm/g722dsp.c
 tests/checkasm/h263dsp.c
 tests/checkasm/h264chroma.c
 tests/checkasm/h264dsp.c
 tests/checkasm/h264pred.c
 tests/checkasm/h264qpel.c
 tests/checkasm/hevc_add_res.c
 tests/checkasm/hevc_deblock.c
 tests/checkasm/hevc_idct.c
 tests/checkasm/hevc_pel.c
 tests/checkasm/hevc_sao.c
 tests/checkasm/huffyuvdsp.c
 tests/checkasm/idctdsp.c
 tests/checkasm/jpeg2000dsp.c
 tests/checkasm/llauddsp.c
 tests/checkasm/lls.c
 tests/checkasm/llviddsp.c
 tests/checkasm/llvidencdsp.c
 tests/checkasm/lpc.c
 tests/checkasm/motion.c
 tests/checkasm/opusdsp.c
 tests/checkasm/pixblockdsp.c
 tests/checkasm/riscv/*
 tests/checkasm/rv34dsp.c
 tests/checkasm/rv40dsp.c
 tests/checkasm/sbrdsp.c
 tests/checkasm/scene_sad.c
 tests/checkasm/svq1enc.c
 tests/checkasm/sw_gbrp.c
 tests/checkasm/sw_range_convert.c
 tests/checkasm/sw_rgb.c
 tests/checkasm/sw_scale.c
 tests/checkasm/sw_yuv2rgb.c
 tests/checkasm/sw_yuv2yuv.c
 tests/checkasm/synth_filter.c
 tests/checkasm/takdsp.c
 tests/checkasm/utvideodsp.c
 tests/checkasm/v210dec.c
 tests/checkasm/v210enc.c
 tests/checkasm/vc1dsp.c
 tests/checkasm/vf_blackdetect.c
 tests/checkasm/vf_blend.c
 tests/checkasm/vf_bwdif.c
 tests/checkasm/vf_colordetect.c
 tests/checkasm/vf_convolution.c
 tests/checkasm/vf_eq.c
 tests/checkasm/vf_gblur.c
 tests/checkasm/vf_hflip.c
 tests/checkasm/vf_threshold.c
 tests/checkasm/videodsp.c
 tests/checkasm/vorbisdsp.c
 tests/checkasm/vp8dsp.c
 tests/checkasm/vp9dsp.c
 tests/checkasm/vvc_alf.c
 tests/checkasm/vvc_mc.c
 tests/checkasm/vvc_sao.c
 tests/checkasm/x86/*
 tests/tiny_ssim.c
 tools/coverity.c
 tools/target_dec_fate.sh
Copyright:
 1989-2001, Free Software Foundation, Inc
 1997-2001, ZSNES Team <zsknight@zsnes.com>
 2000-2002, Fabrice Bellard
 2001-2003, Donald A. Graft
 2001-2018, Michael Niedermayer <michaelni@gmx.at>
 2002, A'rpi
 2002, Jindrich Makovicka <makovick@gmail.com>
 2002-2003, Brian J. Murrell
 2003, Daniel Moreno <comac@comac.darktech.org>
 2003, LeFunGus <lefungus@altern.org>
 2003, Michael Zucchi <notzed@ximian.com>
 2003, Rich Felker
 2003-2004, Tobias Diedrich
 2003-2013, Loren Merritt <lorenm@u.washington.edu>
 2004, Romain Dolbeau <romain@dolbeau.org>
 2004, Ville Saari
 2005, Nikolaj Poroshin <porosh3@psu.ru>
 2006, Ivo van Poorten
 2006, Julian Hall
 2006-2011, Kevin Stone
 2010, Baptiste Coudurier <baptiste.coudurier@free.fr>
 2010, Gordon Schmidt <gordon.schmidt@s2000.tu-chemnitz.de>
 2010, Niel van der Westhuizen <nielkie@gmail.com>
 2010-2012, Stefano Sabatini <stefano.sabatini-lala@poste.it>
 2011, Mans Rullgard <mans@mansr.com>
 2012, Jeremy Tran
 2012-2013, Clément Bœsch <u@pkh.me>
 2012-2015, Henrik Gramner
 2013, Vittorio Giovara <vittorio.giovara@gmail.com>
 2013-2015, Jean Delvare <jdelvare@suse.com>
 2013-2017, Paul B Mahol
 2014, Kieran Kunhya <kierank@obe.tv>
 2014, Red Hat, Inc
 2014-2015, Arwa Arif <arwaarif1994@gmail.com>
 2014-2019, James Darnley <james.darnley@gmail.com>
 2015, Janne Grunau
 2015, Rodger Combs <rodger.combs@gmail.com>
 2015-2016, Martin Storsjo
 2015-2016, Ronald S. Bultje <rsbultje@gmail.com>
 2015-2016, Tiancheng "Timothy" Gu
 2015-2017, James Almer
 2016, Alexandra Hájková
 2017, Gerion Entrup
 2017, Jokyo Images
 2017, Thomas Mundt <tmundt75@gmail.com>
 2018, Two Orioles, LLC
 2018, VideoLAN and dav1d authors
 2018, Yingming Fan <yingmingfan@gmail.com>
 2021, Josh Dekker
 2022, Ben Avison
 2022-2024, Rémi Denis-Courmont
 2023-2024, Institute of Software Chinese Academy of Sciences (ISCAS)
 2023-2024, Nuo Mi <nuomi2021@gmail.com>
 2023-2024, Wu Jianhua <toqsxw@outlook.com>
 2024, Geoff Hill <geoff@geoffhill.org>
 2024, Kyosuke Kawakami
 Lynne
 Wilbert Dijkhof
License: GPL-2+
 FFmpeg is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.
 .
 FFmpeg is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
Comment:
 On Debian systems, the full text of the GNU General Public License
 version 2 can be found in the file `/usr/share/common-licenses/GPL-2'.

Files:
 compat/solaris/*
 doc/t2h.pm
 libavfilter/vf_lensfun.c
Copyright:
 2007, Andrew Zabolotny
 2007-2013, Free Software Foundation, Inc
 2014, Andreas Cadhalpun <Andreas.Cadhalpun@googlemail.com>
 2014, Tiancheng "Timothy" Gu <timothygu99@gmail.com>
 2018, Stephen Seo
License: GPL-3+
 FFmpeg is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.
 .
 FFmpeg is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.
Comment:
 On Debian systems, the full text of the GNU General Public License
 version 3 can be found in the file `/usr/share/common-licenses/GPL-3'.

Files:
 doc/bootstrap.min.css
 doc/examples/avio_http_serve_files.c
 doc/examples/avio_list_dir.c
 doc/examples/avio_read_callback.c
 doc/examples/decode_audio.c
 doc/examples/decode_filter_audio.c
 doc/examples/decode_filter_video.c
 doc/examples/decode_video.c
 doc/examples/demux_decode.c
 doc/examples/encode_audio.c
 doc/examples/encode_video.c
 doc/examples/extract_mvs.c
 doc/examples/hw_decode.c
 doc/examples/mux.c
 doc/examples/qsv_decode.c
 doc/examples/qsv_transcode.c
 doc/examples/remux.c
 doc/examples/resample_audio.c
 doc/examples/scale_video.c
 doc/examples/show_metadata.c
 doc/examples/transcode.c
 doc/examples/vaapi_encode.c
 doc/examples/vaapi_transcode.c
 doc/style.min.css
 ffbuild/bin2c.c
 libavcodec/arm/jrevdct_arm.S
 libavcodec/cinepakenc.c
 libavcodec/nellymoser.c
 libavcodec/nellymoser.h
 libavcodec/nellymoserdec.c
 libavcodec/texturedsp.c
 libavcodec/texturedspenc.c
 libavcodec/x86/vc1dsp_init.c
 libavcodec/x86/vc1dsp_mmx.c
 libavfilter/af_deesser.c
 libavfilter/cuda/vector_helpers.cuh
 libavfilter/vf_avgblur.c
 libavfilter/vf_bilateral.c
 libavfilter/vf_bm3d.c
 libavfilter/vf_colorspace_cuda.c
 libavfilter/vf_colorspace_cuda.cu
 libavfilter/vf_deband.c
 libavfilter/vf_morpho.c
 libavfilter/vf_scale_cuda.c
 libavfilter/vf_scale_cuda.cu
 libavfilter/vf_scale_cuda.h
 libavfilter/vf_thumbnail_cuda.c
 libavfilter/vf_thumbnail_cuda.cu
 libavformat/oggdec.c
 libavformat/oggdec.h
 libavformat/oggparseogm.c
 libavformat/oggparsespeex.c
 libavformat/oggparsetheora.c
 libavformat/oggparsevorbis.c
 libavutil/avsscanf.c
 tests/api/api-band-test.c
 tests/api/api-dump-stream-meta-test.c
 tests/api/api-flac-test.c
 tests/api/api-h264-slice-test.c
 tests/api/api-h264-test.c
 tests/api/api-seek-test.c
 tests/api/api-threadmessage-test.c
 tests/reference.pnm
Copyright:
 2001, Lionel Ulmer <lionel.ulmer@free.fr>
 2001-2003, Fabrice Bellard
 2005, Alex Beregszaszi
 2005, Matthieu CASTET
 2005, Michael Ahlberg
 2005, Måns Rullgård
 2005-2014, Rich Felker
 2007, Benjamin Larsson
 2007, Christophe GISQUET <christophe.gisquet@free.fr>
 2007, Loic Minier <lool@dooz.org>
 2008, Reimar Döffinger <Reimar.Doeffinger@gmx.de>
 2009, Benjamin Dobell
 2009, Glass Echidna
 2010, Nicolas George
 2010-2022, NVIDIA Corporation
 2011, Tomas Härdin
 2011-2014, Reinhard Tartler
 2011-2014, Stefano Sabatini <stefano.sabatini-lala@poste.it>
 2011-2014, Twitter, Inc
 2012, Matthäus G. "Anteru" Chajdas (http://anteru.net)
 2012-2014, Clément Bœsch <u@pkh.me>
 2013-2014, Rl, Aetey Global Technologies AB
 2014, Andrey Utkin
 2014, Barbara Lepage <db0company@gmail.com>
 2014, Lukasz Marek <lukasz.m.luki@gmail.com>
 2015, Anton Khirnov <anton@khirnov.net>
 2015, Ludmila Glinskih
 2015, Matthieu Bouron <matthieu.bouron@stupeflix.com>
 2015, Niklas Haas
 2015, Stephan Holljes
 2015, Vittorio Giovara <vittorio.giovara@gmail.com>
 2015-2016, mawen1250
 2015-2021, Paul B Mahol
 2016, ReneBrals
 2017, Jun Zhao
 2017, Kaixuan Liu
 2017, Ming Yang
 2018, Chris Johnson
 2025, Romain Beauxis
License: Expat

Files:
 libavcodec/aacsbr_fixed.c
 libavcodec/ac3dec_fixed.c
 libavcodec/arm/vp8dsp_armv6.S
 libavcodec/ilbcdata.h
 libavcodec/ilbcdec.c
 libavcodec/mips/ac3dsp_mips.c
 libavcodec/mips/acelp_filters_mips.c
 libavcodec/mips/acelp_vectors_mips.c
 libavcodec/mips/amrwbdec_mips.c
 libavcodec/mips/amrwbdec_mips.h
 libavcodec/mips/celp_filters_mips.c
 libavcodec/mips/celp_math_mips.c
 libavcodec/mips/compute_antialias_fixed.h
 libavcodec/mips/compute_antialias_float.h
 libavcodec/mips/fmtconvert_mips.c
 libavcodec/mips/lsp_mips.h
 libavcodec/mips/mpegaudiodsp_mips_fixed.c
 libavcodec/mips/mpegaudiodsp_mips_float.c
 libavcodec/vvc/itx_1d.c
 libavfilter/opencl/deshake.cl
 libavfilter/vf_deshake_opencl.c
 libavutil/fixed_dsp.c
 libavutil/fixed_dsp.h
 libavutil/mips/float_dsp_mips.c
 libavutil/mips/libm_mips.h
 libavutil/softfloat_tables.h
Copyright:
 2000-2008, Intel Corporation
 2005-2006, Oded Shimon <ods15@ods15.dyndns.org>
 2006-2007, Maxim Gavrilov <maxim.gavrilov@gmail.com>
 2008-2009, Robert Swain <rob@opendot.cl>
 2009, Willow Garage Inc
 2009-2010, Alex Converse <alex.converse@gmail.com>
 2010, Google Inc
 2010, Rob Clark <rob@ti.com>
 2010-2021, ITU/ISO/IEC
 2011, Mans Rullgard <mans@mansr.com>
 2012-2013, MIPS Technologies, Inc., California
 2013, OpenCV Foundation
 2013, The WebRTC project authors
 2023, Nuo Mi
License: LGPL-2.1+ and BSD-3-clause

Files:
 tools/cws2fws.c
 tools/qt-faststart.c
Copyright:
 Alex Beregszaszi
 Mike Melanson <melanson@pcisys.net>
License: public-domain
 This file is placed in the public domain. Use the program however you see fit.

Files:
 compat/windows/makedef
 libavcodec/faandct.c
 libavcodec/interplayacm.c
 libavcodec/libfdk-aacdec.c
 libavcodec/libfdk-aacenc.c
 libavcodec/liblc3dec.c
 libavcodec/liblc3enc.c
 libavcodec/zerocodec.c
 libavdevice/openal-dec.c
 libavfilter/vf_hqx.c
 libavutil/x86/x86inc.asm
Copyright:
 2003, Michael Niedermayer <michaelni@gmx.at>
 2003, Roman Shaposhnik
 2004-2008, Marko Kreen
 2005-2024, x264 project
 2008, Adam Gashlin
 2011, Jonathan Baldwin
 2012, Martin Storsjo
 2012-2013, Derek Buitenhuis
 2014, Clément Bœsch <u@pkh.me>
 2015, Paul B Mahol
 2024, Antoine Soulier <asoulier@google.com>
License: ISC
 Permission to use, copy, modify, and/or distribute this software for any
 purpose with or without fee is hereby granted, provided that the above
 copyright notice and this permission notice appear in all copies.
 .
 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

Files:
 libavcodec/jfdctfst.c
 libavcodec/jfdctint_template.c
 libavcodec/jrevdct.c
Copyright: 1991-1996, Thomas G. Lane
License: IJG
 The authors make NO WARRANTY or representation, either express or implied,
 with respect to this software, its quality, accuracy, merchantability, or
 fitness for a particular purpose.  This software is provided "AS IS", and
 you, its user, assume the entire risk as to its quality and accuracy.
 .
 Permission is hereby granted to use, copy, modify, and distribute this
 software (or portions thereof) for any purpose, without fee, subject to
 these conditions:
 (1) If any part of the source code for this software is distributed, then
 this README file must be included, with this copyright and no-warranty
 notice unaltered; and any additions, deletions, or changes to the original
 files must be clearly indicated in accompanying documentation.
 (2) If only executable code is distributed, then the accompanying
 documentation must state that "this software is based in part on the work
 of the Independent JPEG Group".
 (3) Permission for use of this software is granted only if the user accepts
 full responsibility for any undesirable consequences; the authors accept
 NO LIABILITY for damages of any kind.
 .
 These conditions apply to any software derived from or based on the IJG
 code, not just to the unmodified library.  If you use our work, you ought
 to acknowledge us.
 .
 Permission is NOT granted for the use of any IJG author's name or company
 name in advertising or publicity relating to this software or products
 derived from it.  This software may be referred to only as "the Independent
 JPEG Group's software".
 .
 We specifically permit and encourage the use of this software as the basis
 of commercial products, provided that all warranty or liability claims are
 assumed by the product vendor.

Files:
 libavcodec/aom_film_grain_template.c
 libavcodec/j2kenc.c
Copyright:
 2001-2003, David Janssens
 2002-2003, Yannick Verschueren
 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 2002-2007, Professor Benoit Macq
 2003-2007, Francois-Olivier Devaux and Antonin Descampe
 2005, Herve Drolon, FreeImage Team
 2007, Callum Lerwick <seg@haxxed.com>
 2007, Kamil Nowosad
 2008-2023, Niklas Haas
 2018, Two Orioles, LLC
 2018, VideoLAN and dav1d authors
 2020, Gautam Ramakrishnan <gautamramk@gmail.com>
 Sandflow Consulting LLC
License: LGPL-2.1+ and BSD-2-clause

Files:
 libavcodec/aarch64/fdctdsp_neon.S
 libavutil/adler32.c
Copyright:
 1995, Mark Adler
 2009-2011, Nokia Corporation and/or its subsidiary(-ies)
 2013-2014, Linaro Limited
 2014-2020, D. R. Commander
 2015-2018, Matthieu Darbois
 2016, Siarhei Siamashka
License: Zlib
 This software is provided 'as-is', without any express or implied
 warranty.  In no event will the authors be held liable for any damages
 arising from the use of this software.
 .
 Permission is granted to anyone to use this software for any purpose,
 including commercial applications, and to alter it and redistribute it
 freely, subject to the following restrictions:
 .
 1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software
    in a product, an acknowledgment in the product documentation would be
    appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.
 3. This notice may not be removed or altered from any source distribution.

Files:
 doc/mips.txt
 libavcodec/speexdata.h
 libavcodec/speexdec.c
 libavfilter/af_hdcd.c
Copyright:
 1992-1994, Carsten Bormann
 1992-1994, Jutta Degener
 1993-2006, David Rowe
 2002-2008, Jean-Marc Valin
 2002-2008, Xiph.org Foundation
 2003, EpicGames
 2005-2007, Analog Devices Inc
 2005-2008, Commonwealth Scientific and Industrial Research Organisation (CSIRO)
 2010, Chris Moeller
 2012, MIPS Technologies, Inc., California
License: BSD-3-clause

Files:
 libavfilter/ebur128.c
 libavfilter/ebur128.h
Copyright: 2011, Jan Kokemüller
License: LGPL-2.1+ and Expat

Files:
 libavcodec/riscv/h264addpx_rvv.S
 libavcodec/riscv/h264dsp_rvv.S
 libavcodec/riscv/h264idct_rvv.S
 libavcodec/riscv/h264qpel_rvv.S
 libavcodec/riscv/startcode_rvb.S
 libavcodec/riscv/startcode_rvv.S
 libavfilter/af_arnndn.c
 libavfilter/gblur.h
 libavfilter/vf_gblur.c
 libavfilter/vf_gblur_init.h
Copyright:
 2005-2009, Xiph.Org Foundation
 2007-2008, CSIRO
 2008-2011, Octasic Inc
 2011, Pascal Getreuer
 2016-2019, Paul B Mahol
 2017, Mozilla
 2018, Gregor Richards
 2024, J. Dekker <jdek@itanimul.li>
 2024, Niklas Haas
 2024, Rémi Denis-Courmont
 Jean-Marc Valin
License: BSD-2-clause

Files: libavformat/aadec.c
Copyright:
 2001-2014, Jim Teeuwen
 2015, Vesselin Bontchev
License: BSD-1-clause
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 .
 THIS SOFTWARE IS PROVIDED BY THE MIPS TECHNOLOGIES, INC. ``AS IS'' AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.

Files: libavcodec/libzvbi-teletextdec.c
Copyright:
 2005-2012, Wolfram Gloger
 2013, Marton Balint
License: LGPL-2+
 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2 of the License, or (at your option) any later version.
 .
 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.
Comment:
 On Debian systems, the full text of the GNU Lesser General Public License
 version 2 can be found in the file `/usr/share/common-licenses/GPL-2'.

Files: libswresample/resample.c
Copyright:
 2004-2012, Michael Niedermayer <michaelni@gmx.at>
 2006, Xiaogang Zhang
License: LGPL-2.1+ and BSL

Files:
 libavformat/imf.h
 libavformat/imf_cpl.c
 libavformat/imfdec.c
 libavformat/tests/imf.c
Copyright: Sandflow Consulting LLC
License: BSD-2-clause or LGPL-2.1+

Files:
 libavcodec/jpeg2000htdec.c
 libavutil/uuid.c
Copyright:
 1996-1997, Theodore Ts'o
 2019-2021, Osamu Watanabe
 2022, Caleb Etemesi <etemesicaleb@gmail.com>
 2022, Pierre-Anthony Lemieux <pal@palemieux.com>
 Zane van Iperen <zane@zanevaniperen.com>
License: BSD-3-clause or LGPL-2.1+

Files:
 libavutil/libm.h
 libavutil/mathematics.c
Copyright:
 2005-2012, Michael Niedermayer <michaelni@gmx.at>
 2006, John Maddock
License: BSL or LGPL-2.1+

Files: debian/*
Copyright:
 2014-2017, Andreas Cadhalpun <Andreas.Cadhalpun@googlemail.com>
 2016-2025, Sebastian Ramacher <sramacher@debian.org>
 2017-2018, James Cowgill <jcowgill@debian.org>
 2020, Jonas Smedegaard <dr@jones.dk>
License: LGPL-2.1+

Files: debian/missing-sources/*
Copyright: 2014, Barbara Lepage <db0company@gmail.com>
License: Expat
Comment:
 This is the source of doc/style.min.css.
 It was obtained from the FFmpeg web repository:
 <https://github.com/FFmpeg/web>

Files: debian/qt-faststart.1
Copyright: Andres Mejia <mcitadel@gmail.com>
License: man-page
 This manual page was written by Andres Mejia <mcitadel@gmail.com>
 for the Debian GNU/Linux system, but may be used by others.

License: BSD-2-clause
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 .
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.

License: BSD-3-clause
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. Neither the name of the MIPS Technologies, Inc., nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.
 .
 THIS SOFTWARE IS PROVIDED BY THE MIPS TECHNOLOGIES, INC. ``AS IS'' AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED.  IN NO EVENT SHALL THE MIPS TECHNOLOGIES, INC. BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 SUCH DAMAGE.

License: BSL
 Permission is hereby granted, free of charge, to any person or organization
 obtaining a copy of the software and accompanying documentation covered by
 this license (the "Software") to use, reproduce, display, distribute,
 execute, and transmit the Software, and to prepare derivative works of the
 Software, and to permit third-parties to whom the Software is furnished to
 do so, all subject to the following:
 .
 The copyright notices in the Software and this entire statement, including
 the above license grant, this restriction and the following disclaimer,
 must be included in all copies of the Software, in whole or in part, and
 all derivative works of the Software, unless such copies or derivative
 works are solely in the form of machine-executable object code generated by
 a source language processor.
 .
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.

License: Expat
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 .
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 .
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.

License: LGPL-2.1+
 FFmpeg is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.
 .
 FFmpeg is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.
Comment:
 On Debian systems, the full text of the GNU Lesser General Public License
 version 2.1 can be found in the file `/usr/share/common-licenses/LGPL-2.1'.

License: Sundry
 This source code is freely redistributable and may be used for
 any purpose.  This copyright notice must be maintained.
 Juergen Mueller/Edward Beingessner And Sundry Contributors are not responsible for
 the consequences of using this software.
```

### SDL

Copyright:
- 1997-2026 Sam Lantinga <slouken@libsdl.org>

License: Zlib

### SDL_mixer

Copyright:
- 1997-2026 Sam Lantinga <slouken@libsdl.org>

License: Zlib

### SDL_ttf

Copyright:
- 1997-2012 Sam Lantinga <slouken@libsdl.org>

License: Zlib

A companion library to SDL for working with TrueType (tm) fonts.

### SpeexDSP

Copyright:
- 2002-2008 	Xiph.org Foundation
- 2002-2008 	Jean-Marc Valin
- 2005-2007	Analog Devices Inc.
- 2005-2008	Commonwealth Scientific and Industrial Research
                          Organisation (CSIRO)
- 1993, 2002, 2006 David Rowe
- 2003 		EpicGames
- 1992-1994	Jutta Degener, Carsten Bormann

License: BSD-3-Clause

### zfec

Copyright:
-  1997-98 Luigi Rizzo (luigi@iet.unipi.it)

License: GPL-2+ or TGPPL-1.0

Portions derived from code by Phil Karn (karn@ka9q.ampr.org),
Robert Morelos-Zaragoza (robert@spectra.eng.hawaii.edu) and Hari
Thirumoorthy (harit@spectra.eng.hawaii.edu), Aug 1995

Modifications by Dan Rubenstein (see Modifications.txt for
their description.
Modifications (C) 1998 Dan Rubenstein (drubenst@cs.umass.edu)

Licenses
--------

### BSD-2-clause

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### BSD-3-clause

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the <copyright holder> nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY <COPYRIGHT HOLDER> AS IS AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### BSD-4-clause

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software
   must display the following acknowledgement: This product includes software
   developed by the <copyright holder>.

4. Neither the name of the <copyright holder> nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY <COPYRIGHT HOLDER> AS IS AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### GPL-2+

```
                    GNU GENERAL PUBLIC LICENSE
                       Version 2, June 1991

 Copyright (C) 1989, 1991 Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The licenses for most software are designed to take away your
freedom to share and change it.  By contrast, the GNU General Public
License is intended to guarantee your freedom to share and change free
software--to make sure the software is free for all its users.  This
General Public License applies to most of the Free Software
Foundation's software and to any other program whose authors commit to
using it.  (Some other Free Software Foundation software is covered by
the GNU Lesser General Public License instead.)  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
this service if you wish), that you receive source code or can get it
if you want it, that you can change the software or use pieces of it
in new free programs; and that you know you can do these things.

  To protect your rights, we need to make restrictions that forbid
anyone to deny you these rights or to ask you to surrender the rights.
These restrictions translate to certain responsibilities for you if you
distribute copies of the software, or if you modify it.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must give the recipients all the rights that
you have.  You must make sure that they, too, receive or can get the
source code.  And you must show them these terms so they know their
rights.

  We protect your rights with two steps: (1) copyright the software, and
(2) offer you this license which gives you legal permission to copy,
distribute and/or modify the software.

  Also, for each author's protection and ours, we want to make certain
that everyone understands that there is no warranty for this free
software.  If the software is modified by someone else and passed on, we
want its recipients to know that what they have is not the original, so
that any problems introduced by others will not reflect on the original
authors' reputations.

  Finally, any free program is threatened constantly by software
patents.  We wish to avoid the danger that redistributors of a free
program will individually obtain patent licenses, in effect making the
program proprietary.  To prevent this, we have made it clear that any
patent must be licensed for everyone's free use or not licensed at all.

  The precise terms and conditions for copying, distribution and
modification follow.

                    GNU GENERAL PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. This License applies to any program or other work which contains
a notice placed by the copyright holder saying it may be distributed
under the terms of this General Public License.  The "Program", below,
refers to any such program or work, and a "work based on the Program"
means either the Program or any derivative work under copyright law:
that is to say, a work containing the Program or a portion of it,
either verbatim or with modifications and/or translated into another
language.  (Hereinafter, translation is included without limitation in
the term "modification".)  Each licensee is addressed as "you".

Activities other than copying, distribution and modification are not
covered by this License; they are outside its scope.  The act of
running the Program is not restricted, and the output from the Program
is covered only if its contents constitute a work based on the
Program (independent of having been made by running the Program).
Whether that is true depends on what the Program does.

  1. You may copy and distribute verbatim copies of the Program's
source code as you receive it, in any medium, provided that you
conspicuously and appropriately publish on each copy an appropriate
copyright notice and disclaimer of warranty; keep intact all the
notices that refer to this License and to the absence of any warranty;
and give any other recipients of the Program a copy of this License
along with the Program.

You may charge a fee for the physical act of transferring a copy, and
you may at your option offer warranty protection in exchange for a fee.

  2. You may modify your copy or copies of the Program or any portion
of it, thus forming a work based on the Program, and copy and
distribute such modifications or work under the terms of Section 1
above, provided that you also meet all of these conditions:

    a) You must cause the modified files to carry prominent notices
    stating that you changed the files and the date of any change.

    b) You must cause any work that you distribute or publish, that in
    whole or in part contains or is derived from the Program or any
    part thereof, to be licensed as a whole at no charge to all third
    parties under the terms of this License.

    c) If the modified program normally reads commands interactively
    when run, you must cause it, when started running for such
    interactive use in the most ordinary way, to print or display an
    announcement including an appropriate copyright notice and a
    notice that there is no warranty (or else, saying that you provide
    a warranty) and that users may redistribute the program under
    these conditions, and telling the user how to view a copy of this
    License.  (Exception: if the Program itself is interactive but
    does not normally print such an announcement, your work based on
    the Program is not required to print an announcement.)

These requirements apply to the modified work as a whole.  If
identifiable sections of that work are not derived from the Program,
and can be reasonably considered independent and separate works in
themselves, then this License, and its terms, do not apply to those
sections when you distribute them as separate works.  But when you
distribute the same sections as part of a whole which is a work based
on the Program, the distribution of the whole must be on the terms of
this License, whose permissions for other licensees extend to the
entire whole, and thus to each and every part regardless of who wrote it.

Thus, it is not the intent of this section to claim rights or contest
your rights to work written entirely by you; rather, the intent is to
exercise the right to control the distribution of derivative or
collective works based on the Program.

In addition, mere aggregation of another work not based on the Program
with the Program (or with a work based on the Program) on a volume of
a storage or distribution medium does not bring the other work under
the scope of this License.

  3. You may copy and distribute the Program (or a work based on it,
under Section 2) in object code or executable form under the terms of
Sections 1 and 2 above provided that you also do one of the following:

    a) Accompany it with the complete corresponding machine-readable
    source code, which must be distributed under the terms of Sections
    1 and 2 above on a medium customarily used for software interchange; or,

    b) Accompany it with a written offer, valid for at least three
    years, to give any third party, for a charge no more than your
    cost of physically performing source distribution, a complete
    machine-readable copy of the corresponding source code, to be
    distributed under the terms of Sections 1 and 2 above on a medium
    customarily used for software interchange; or,

    c) Accompany it with the information you received as to the offer
    to distribute corresponding source code.  (This alternative is
    allowed only for noncommercial distribution and only if you
    received the program in object code or executable form with such
    an offer, in accord with Subsection b above.)

The source code for a work means the preferred form of the work for
making modifications to it.  For an executable work, complete source
code means all the source code for all modules it contains, plus any
associated interface definition files, plus the scripts used to
control compilation and installation of the executable.  However, as a
special exception, the source code distributed need not include
anything that is normally distributed (in either source or binary
form) with the major components (compiler, kernel, and so on) of the
operating system on which the executable runs, unless that component
itself accompanies the executable.

If distribution of executable or object code is made by offering
access to copy from a designated place, then offering equivalent
access to copy the source code from the same place counts as
distribution of the source code, even though third parties are not
compelled to copy the source along with the object code.

  4. You may not copy, modify, sublicense, or distribute the Program
except as expressly provided under this License.  Any attempt
otherwise to copy, modify, sublicense or distribute the Program is
void, and will automatically terminate your rights under this License.
However, parties who have received copies, or rights, from you under
this License will not have their licenses terminated so long as such
parties remain in full compliance.

  5. You are not required to accept this License, since you have not
signed it.  However, nothing else grants you permission to modify or
distribute the Program or its derivative works.  These actions are
prohibited by law if you do not accept this License.  Therefore, by
modifying or distributing the Program (or any work based on the
Program), you indicate your acceptance of this License to do so, and
all its terms and conditions for copying, distributing or modifying
the Program or works based on it.

  6. Each time you redistribute the Program (or any work based on the
Program), the recipient automatically receives a license from the
original licensor to copy, distribute or modify the Program subject to
these terms and conditions.  You may not impose any further
restrictions on the recipients' exercise of the rights granted herein.
You are not responsible for enforcing compliance by third parties to
this License.

  7. If, as a consequence of a court judgment or allegation of patent
infringement or for any other reason (not limited to patent issues),
conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot
distribute so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you
may not distribute the Program at all.  For example, if a patent
license would not permit royalty-free redistribution of the Program by
all those who receive copies directly or indirectly through you, then
the only way you could satisfy both it and this License would be to
refrain entirely from distribution of the Program.

If any portion of this section is held invalid or unenforceable under
any particular circumstance, the balance of the section is intended to
apply and the section as a whole is intended to apply in other
circumstances.

It is not the purpose of this section to induce you to infringe any
patents or other property right claims or to contest validity of any
such claims; this section has the sole purpose of protecting the
integrity of the free software distribution system, which is
implemented by public license practices.  Many people have made
generous contributions to the wide range of software distributed
through that system in reliance on consistent application of that
system; it is up to the author/donor to decide if he or she is willing
to distribute software through any other system and a licensee cannot
impose that choice.

This section is intended to make thoroughly clear what is believed to
be a consequence of the rest of this License.

  8. If the distribution and/or use of the Program is restricted in
certain countries either by patents or by copyrighted interfaces, the
original copyright holder who places the Program under this License
may add an explicit geographical distribution limitation excluding
those countries, so that distribution is permitted only in or among
countries not thus excluded.  In such case, this License incorporates
the limitation as if written in the body of this License.

  9. The Free Software Foundation may publish revised and/or new versions
of the General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

Each version is given a distinguishing version number.  If the Program
specifies a version number of this License which applies to it and "any
later version", you have the option of following the terms and conditions
either of that version or of any later version published by the Free
Software Foundation.  If the Program does not specify a version number of
this License, you may choose any version ever published by the Free Software
Foundation.

  10. If you wish to incorporate parts of the Program into other free
programs whose distribution conditions are different, write to the author
to ask for permission.  For software which is copyrighted by the Free
Software Foundation, write to the Free Software Foundation; we sometimes
make exceptions for this.  Our decision will be guided by the two goals
of preserving the free status of all derivatives of our free software and
of promoting the sharing and reuse of software generally.

                            NO WARRANTY

  11. BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO WARRANTY
FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW.  EXCEPT WHEN
OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES
PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE ENTIRE RISK AS
TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU.  SHOULD THE
PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING,
REPAIR OR CORRECTION.

  12. IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MAY MODIFY AND/OR
REDISTRIBUTE THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES,
INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING
OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED
TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY
YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER
PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
convey the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Also add information on how to contact you by electronic and paper mail.

If the program is interactive, make it output a short notice like this
when it starts in an interactive mode:

    Gnomovision version 69, Copyright (C) year name of author
    Gnomovision comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, the commands you use may
be called something other than `show w' and `show c'; they could even be
mouse-clicks or menu items--whatever suits your program.

You should also get your employer (if you work as a programmer) or your
school, if any, to sign a "copyright disclaimer" for the program, if
necessary.  Here is a sample; alter the names:

  Yoyodyne, Inc., hereby disclaims all copyright interest in the program
  `Gnomovision' (which makes passes at compilers) written by James Hacker.

  <signature of Ty Coon>, 1 April 1989
  Ty Coon, President of Vice

This General Public License does not permit incorporating your program into
proprietary programs.  If your program is a subroutine library, you may
consider it more useful to permit linking proprietary applications with the
library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.
```

### GPL-3
```

                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
software and other kinds of works.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
the GNU General Public License is intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.  We, the Free Software Foundation, use the
GNU General Public License for most of our software; it applies also to
any other work released this way by its authors.  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  To protect your rights, we need to prevent others from denying you
these rights or asking you to surrender the rights.  Therefore, you have
certain responsibilities if you distribute copies of the software, or if
you modify it: responsibilities to respect the freedom of others.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must pass on to the recipients the same
freedoms that you received.  You must make sure that they, too, receive
or can get the source code.  And you must show them these terms so they
know their rights.

  Developers that use the GNU GPL protect your rights with two steps:
(1) assert copyright on the software, and (2) offer you this License
giving you legal permission to copy, distribute and/or modify it.

  For the developers' and authors' protection, the GPL clearly explains
that there is no warranty for this free software.  For both users' and
authors' sake, the GPL requires that modified versions be marked as
changed, so that their problems will not be attributed erroneously to
authors of previous versions.

  Some devices are designed to deny users access to install or run
modified versions of the software inside them, although the manufacturer
can do so.  This is fundamentally incompatible with the aim of
protecting users' freedom to change the software.  The systematic
pattern of such abuse occurs in the area of products for individuals to
use, which is precisely where it is most unacceptable.  Therefore, we
have designed this version of the GPL to prohibit the practice for those
products.  If such problems arise substantially in other domains, we
stand ready to extend this provision to those domains in future versions
of the GPL, as needed to protect the freedom of users.

  Finally, every program is threatened constantly by software patents.
States should not allow patents to restrict development and use of
software on general-purpose computers, but in those that do, we wish to
avoid the special danger that patents applied to a free program could
make it effectively proprietary.  To prevent this, the GPL assures that
patents cannot be used to render the program non-free.

  The precise terms and conditions for copying, distribution and
modification follow.

                       TERMS AND CONDITIONS

  0. Definitions.

  "This License" refers to version 3 of the GNU General Public License.

  "Copyright" also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

  "The Program" refers to any copyrightable work licensed under this
License.  Each licensee is addressed as "you".  "Licensees" and
"recipients" may be individuals or organizations.

  To "modify" a work means to copy from or adapt all or part of the work
in a fashion requiring copyright permission, other than the making of an
exact copy.  The resulting work is called a "modified version" of the
earlier work or a work "based on" the earlier work.

  A "covered work" means either the unmodified Program or a work based
on the Program.

  To "propagate" a work means to do anything with it that, without
permission, would make you directly or secondarily liable for
infringement under applicable copyright law, except executing it on a
computer or modifying a private copy.  Propagation includes copying,
distribution (with or without modification), making available to the
public, and in some countries other activities as well.

  To "convey" a work means any kind of propagation that enables other
parties to make or receive copies.  Mere interaction with a user through
a computer network, with no transfer of a copy, is not conveying.

  An interactive user interface displays "Appropriate Legal Notices"
to the extent that it includes a convenient and prominently visible
feature that (1) displays an appropriate copyright notice, and (2)
tells the user that there is no warranty for the work (except to the
extent that warranties are provided), that licensees may convey the
work under this License, and how to view a copy of this License.  If
the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

  1. Source Code.

  The "source code" for a work means the preferred form of the work
for making modifications to it.  "Object code" means any non-source
form of a work.

  A "Standard Interface" means an interface that either is an official
standard defined by a recognized standards body, or, in the case of
interfaces specified for a particular programming language, one that
is widely used among developers working in that language.

  The "System Libraries" of an executable work include anything, other
than the work as a whole, that (a) is included in the normal form of
packaging a Major Component, but which is not part of that Major
Component, and (b) serves only to enable use of the work with that
Major Component, or to implement a Standard Interface for which an
implementation is available to the public in source code form.  A
"Major Component", in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system
(if any) on which the executable work runs, or a compiler used to
produce the work, or an object code interpreter used to run it.

  The "Corresponding Source" for a work in object code form means all
the source code needed to generate, install, and (for an executable
work) run the object code and to modify the work, including scripts to
control those activities.  However, it does not include the work's
System Libraries, or general-purpose tools or generally available free
programs which are used unmodified in performing those activities but
which are not part of the work.  For example, Corresponding Source
includes interface definition files associated with source files for
the work, and the source code for shared libraries and dynamically
linked subprograms that the work is specifically designed to require,
such as by intimate data communication or control flow between those
subprograms and other parts of the work.

  The Corresponding Source need not include anything that users
can regenerate automatically from other parts of the Corresponding
Source.

  The Corresponding Source for a work in source code form is that
same work.

  2. Basic Permissions.

  All rights granted under this License are granted for the term of
copyright on the Program, and are irrevocable provided the stated
conditions are met.  This License explicitly affirms your unlimited
permission to run the unmodified Program.  The output from running a
covered work is covered by this License only if the output, given its
content, constitutes a covered work.  This License acknowledges your
rights of fair use or other equivalent, as provided by copyright law.

  You may make, run and propagate covered works that you do not
convey, without conditions so long as your license otherwise remains
in force.  You may convey covered works to others for the sole purpose
of having them make modifications exclusively for you, or provide you
with facilities for running those works, provided that you comply with
the terms of this License in conveying all material for which you do
not control copyright.  Those thus making or running the covered works
for you must do so exclusively on your behalf, under your direction
and control, on terms that prohibit them from making any copies of
your copyrighted material outside their relationship with you.

  Conveying under any other circumstances is permitted solely under
the conditions stated below.  Sublicensing is not allowed; section 10
makes it unnecessary.

  3. Protecting Users' Legal Rights From Anti-Circumvention Law.

  No covered work shall be deemed part of an effective technological
measure under any applicable law fulfilling obligations under article
11 of the WIPO copyright treaty adopted on 20 December 1996, or
similar laws prohibiting or restricting circumvention of such
measures.

  When you convey a covered work, you waive any legal power to forbid
circumvention of technological measures to the extent such circumvention
is effected by exercising rights under this License with respect to
the covered work, and you disclaim any intention to limit operation or
modification of the work as a means of enforcing, against the work's
users, your or third parties' legal rights to forbid circumvention of
technological measures.

  4. Conveying Verbatim Copies.

  You may convey verbatim copies of the Program's source code as you
receive it, in any medium, provided that you conspicuously and
appropriately publish on each copy an appropriate copyright notice;
keep intact all notices stating that this License and any
non-permissive terms added in accord with section 7 apply to the code;
keep intact all notices of the absence of any warranty; and give all
recipients a copy of this License along with the Program.

  You may charge any price or no price for each copy that you convey,
and you may offer support or warranty protection for a fee.

  5. Conveying Modified Source Versions.

  You may convey a work based on the Program, or the modifications to
produce it from the Program, in the form of source code under the
terms of section 4, provided that you also meet all of these conditions:

    a) The work must carry prominent notices stating that you modified
    it, and giving a relevant date.

    b) The work must carry prominent notices stating that it is
    released under this License and any conditions added under section
    7.  This requirement modifies the requirement in section 4 to
    "keep intact all notices".

    c) You must license the entire work, as a whole, under this
    License to anyone who comes into possession of a copy.  This
    License will therefore apply, along with any applicable section 7
    additional terms, to the whole of the work, and all its parts,
    regardless of how they are packaged.  This License gives no
    permission to license the work in any other way, but it does not
    invalidate such permission if you have separately received it.

    d) If the work has interactive user interfaces, each must display
    Appropriate Legal Notices; however, if the Program has interactive
    interfaces that do not display Appropriate Legal Notices, your
    work need not make them do so.

  A compilation of a covered work with other separate and independent
works, which are not by their nature extensions of the covered work,
and which are not combined with it such as to form a larger program,
in or on a volume of a storage or distribution medium, is called an
"aggregate" if the compilation and its resulting copyright are not
used to limit the access or legal rights of the compilation's users
beyond what the individual works permit.  Inclusion of a covered work
in an aggregate does not cause this License to apply to the other
parts of the aggregate.

  6. Conveying Non-Source Forms.

  You may convey a covered work in object code form under the terms
of sections 4 and 5, provided that you also convey the
machine-readable Corresponding Source under the terms of this License,
in one of these ways:

    a) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by the
    Corresponding Source fixed on a durable physical medium
    customarily used for software interchange.

    b) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by a
    written offer, valid for at least three years and valid for as
    long as you offer spare parts or customer support for that product
    model, to give anyone who possesses the object code either (1) a
    copy of the Corresponding Source for all the software in the
    product that is covered by this License, on a durable physical
    medium customarily used for software interchange, for a price no
    more than your reasonable cost of physically performing this
    conveying of source, or (2) access to copy the
    Corresponding Source from a network server at no charge.

    c) Convey individual copies of the object code with a copy of the
    written offer to provide the Corresponding Source.  This
    alternative is allowed only occasionally and noncommercially, and
    only if you received the object code with such an offer, in accord
    with subsection 6b.

    d) Convey the object code by offering access from a designated
    place (gratis or for a charge), and offer equivalent access to the
    Corresponding Source in the same way through the same place at no
    further charge.  You need not require recipients to copy the
    Corresponding Source along with the object code.  If the place to
    copy the object code is a network server, the Corresponding Source
    may be on a different server (operated by you or a third party)
    that supports equivalent copying facilities, provided you maintain
    clear directions next to the object code saying where to find the
    Corresponding Source.  Regardless of what server hosts the
    Corresponding Source, you remain obligated to ensure that it is
    available for as long as needed to satisfy these requirements.

    e) Convey the object code using peer-to-peer transmission, provided
    you inform other peers where the object code and Corresponding
    Source of the work are being offered to the general public at no
    charge under subsection 6d.

  A separable portion of the object code, whose source code is excluded
from the Corresponding Source as a System Library, need not be
included in conveying the object code work.

  A "User Product" is either (1) a "consumer product", which means any
tangible personal property which is normally used for personal, family,
or household purposes, or (2) anything designed or sold for incorporation
into a dwelling.  In determining whether a product is a consumer product,
doubtful cases shall be resolved in favor of coverage.  For a particular
product received by a particular user, "normally used" refers to a
typical or common use of that class of product, regardless of the status
of the particular user or of the way in which the particular user
actually uses, or expects or is expected to use, the product.  A product
is a consumer product regardless of whether the product has substantial
commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

  "Installation Information" for a User Product means any methods,
procedures, authorization keys, or other information required to install
and execute modified versions of a covered work in that User Product from
a modified version of its Corresponding Source.  The information must
suffice to ensure that the continued functioning of the modified object
code is in no case prevented or interfered with solely because
modification has been made.

  If you convey an object code work under this section in, or with, or
specifically for use in, a User Product, and the conveying occurs as
part of a transaction in which the right of possession and use of the
User Product is transferred to the recipient in perpetuity or for a
fixed term (regardless of how the transaction is characterized), the
Corresponding Source conveyed under this section must be accompanied
by the Installation Information.  But this requirement does not apply
if neither you nor any third party retains the ability to install
modified object code on the User Product (for example, the work has
been installed in ROM).

  The requirement to provide Installation Information does not include a
requirement to continue to provide support service, warranty, or updates
for a work that has been modified or installed by the recipient, or for
the User Product in which it has been modified or installed.  Access to a
network may be denied when the modification itself materially and
adversely affects the operation of the network or violates the rules and
protocols for communication across the network.

  Corresponding Source conveyed, and Installation Information provided,
in accord with this section must be in a format that is publicly
documented (and with an implementation available to the public in
source code form), and must require no special password or key for
unpacking, reading or copying.

  7. Additional Terms.

  "Additional permissions" are terms that supplement the terms of this
License by making exceptions from one or more of its conditions.
Additional permissions that are applicable to the entire Program shall
be treated as though they were included in this License, to the extent
that they are valid under applicable law.  If additional permissions
apply only to part of the Program, that part may be used separately
under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

  When you convey a copy of a covered work, you may at your option
remove any additional permissions from that copy, or from any part of
it.  (Additional permissions may be written to require their own
removal in certain cases when you modify the work.)  You may place
additional permissions on material, added by you to a covered work,
for which you have or can give appropriate copyright permission.

  Notwithstanding any other provision of this License, for material you
add to a covered work, you may (if authorized by the copyright holders of
that material) supplement the terms of this License with terms:

    a) Disclaiming warranty or limiting liability differently from the
    terms of sections 15 and 16 of this License; or

    b) Requiring preservation of specified reasonable legal notices or
    author attributions in that material or in the Appropriate Legal
    Notices displayed by works containing it; or

    c) Prohibiting misrepresentation of the origin of that material, or
    requiring that modified versions of such material be marked in
    reasonable ways as different from the original version; or

    d) Limiting the use for publicity purposes of names of licensors or
    authors of the material; or

    e) Declining to grant rights under trademark law for use of some
    trade names, trademarks, or service marks; or

    f) Requiring indemnification of licensors and authors of that
    material by anyone who conveys the material (or modified versions of
    it) with contractual assumptions of liability to the recipient, for
    any liability that these contractual assumptions directly impose on
    those licensors and authors.

  All other non-permissive additional terms are considered "further
restrictions" within the meaning of section 10.  If the Program as you
received it, or any part of it, contains a notice stating that it is
governed by this License along with a term that is a further
restriction, you may remove that term.  If a license document contains
a further restriction but permits relicensing or conveying under this
License, you may add to a covered work material governed by the terms
of that license document, provided that the further restriction does
not survive such relicensing or conveying.

  If you add terms to a covered work in accord with this section, you
must place, in the relevant source files, a statement of the
additional terms that apply to those files, or a notice indicating
where to find the applicable terms.

  Additional terms, permissive or non-permissive, may be stated in the
form of a separately written license, or stated as exceptions;
the above requirements apply either way.

  8. Termination.

  You may not propagate or modify a covered work except as expressly
provided under this License.  Any attempt otherwise to propagate or
modify it is void, and will automatically terminate your rights under
this License (including any patent licenses granted under the third
paragraph of section 11).

  However, if you cease all violation of this License, then your
license from a particular copyright holder is reinstated (a)
provisionally, unless and until the copyright holder explicitly and
finally terminates your license, and (b) permanently, if the copyright
holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

  Moreover, your license from a particular copyright holder is
reinstated permanently if the copyright holder notifies you of the
violation by some reasonable means, this is the first time you have
received notice of violation of this License (for any work) from that
copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

  Termination of your rights under this section does not terminate the
licenses of parties who have received copies or rights from you under
this License.  If your rights have been terminated and not permanently
reinstated, you do not qualify to receive new licenses for the same
material under section 10.

  9. Acceptance Not Required for Having Copies.

  You are not required to accept this License in order to receive or
run a copy of the Program.  Ancillary propagation of a covered work
occurring solely as a consequence of using peer-to-peer transmission
to receive a copy likewise does not require acceptance.  However,
nothing other than this License grants you permission to propagate or
modify any covered work.  These actions infringe copyright if you do
not accept this License.  Therefore, by modifying or propagating a
covered work, you indicate your acceptance of this License to do so.

  10. Automatic Licensing of Downstream Recipients.

  Each time you convey a covered work, the recipient automatically
receives a license from the original licensors, to run, modify and
propagate that work, subject to this License.  You are not responsible
for enforcing compliance by third parties with this License.

  An "entity transaction" is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an
organization, or merging organizations.  If propagation of a covered
work results from an entity transaction, each party to that
transaction who receives a copy of the work also receives whatever
licenses to the work the party's predecessor in interest had or could
give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if
the predecessor has it or can get it with reasonable efforts.

  You may not impose any further restrictions on the exercise of the
rights granted or affirmed under this License.  For example, you may
not impose a license fee, royalty, or other charge for exercise of
rights granted under this License, and you may not initiate litigation
(including a cross-claim or counterclaim in a lawsuit) alleging that
any patent claim is infringed by making, using, selling, offering for
sale, or importing the Program or any portion of it.

  11. Patents.

  A "contributor" is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based.  The
work thus licensed is called the contributor's "contributor version".

  A contributor's "essential patent claims" are all patent claims
owned or controlled by the contributor, whether already acquired or
hereafter acquired, that would be infringed by some manner, permitted
by this License, of making, using, or selling its contributor version,
but do not include claims that would be infringed only as a
consequence of further modification of the contributor version.  For
purposes of this definition, "control" includes the right to grant
patent sublicenses in a manner consistent with the requirements of
this License.

  Each contributor grants you a non-exclusive, worldwide, royalty-free
patent license under the contributor's essential patent claims, to
make, use, sell, offer for sale, import and otherwise run, modify and
propagate the contents of its contributor version.

  In the following three paragraphs, a "patent license" is any express
agreement or commitment, however denominated, not to enforce a patent
(such as an express permission to practice a patent or covenant not to
sue for patent infringement).  To "grant" such a patent license to a
party means to make such an agreement or commitment not to enforce a
patent against the party.

  If you convey a covered work, knowingly relying on a patent license,
and the Corresponding Source of the work is not available for anyone
to copy, free of charge and under the terms of this License, through a
publicly available network server or other readily accessible means,
then you must either (1) cause the Corresponding Source to be so
available, or (2) arrange to deprive yourself of the benefit of the
patent license for this particular work, or (3) arrange, in a manner
consistent with the requirements of this License, to extend the patent
license to downstream recipients.  "Knowingly relying" means you have
actual knowledge that, but for the patent license, your conveying the
covered work in a country, or your recipient's use of the covered work
in a country, would infringe one or more identifiable patents in that
country that you have reason to believe are valid.

  If, pursuant to or in connection with a single transaction or
arrangement, you convey, or propagate by procuring conveyance of, a
covered work, and grant a patent license to some of the parties
receiving the covered work authorizing them to use, propagate, modify
or convey a specific copy of the covered work, then the patent license
you grant is automatically extended to all recipients of the covered
work and works based on it.

  A patent license is "discriminatory" if it does not include within
the scope of its coverage, prohibits the exercise of, or is
conditioned on the non-exercise of one or more of the rights that are
specifically granted under this License.  You may not convey a covered
work if you are a party to an arrangement with a third party that is
in the business of distributing software, under which you make payment
to the third party based on the extent of your activity of conveying
the work, and under which the third party grants, to any of the
parties who would receive the covered work from you, a discriminatory
patent license (a) in connection with copies of the covered work
conveyed by you (or copies made from those copies), or (b) primarily
for and in connection with specific products or compilations that
contain the covered work, unless you entered into that arrangement,
or that patent license was granted, prior to 28 March 2007.

  Nothing in this License shall be construed as excluding or limiting
any implied license or other defenses to infringement that may
otherwise be available to you under applicable patent law.

  12. No Surrender of Others' Freedom.

  If conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot convey a
covered work so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you may
not convey it at all.  For example, if you agree to terms that obligate you
to collect a royalty for further conveying from those to whom you convey
the Program, the only way you could satisfy both those terms and this
License would be to refrain entirely from conveying the Program.

  13. Use with the GNU Affero General Public License.

  Notwithstanding any other provision of this License, you have
permission to link or combine any covered work with a work licensed
under version 3 of the GNU Affero General Public License into a single
combined work, and to convey the resulting work.  The terms of this
License will continue to apply to the part which is the covered work,
but the special requirements of the GNU Affero General Public License,
section 13, concerning interaction through a network will apply to the
combination as such.

  14. Revised Versions of this License.

  The Free Software Foundation may publish revised and/or new versions of
the GNU General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

  Each version is given a distinguishing version number.  If the
Program specifies that a certain numbered version of the GNU General
Public License "or any later version" applies to it, you have the
option of following the terms and conditions either of that numbered
version or of any later version published by the Free Software
Foundation.  If the Program does not specify a version number of the
GNU General Public License, you may choose any version ever published
by the Free Software Foundation.

  If the Program specifies that a proxy can decide which future
versions of the GNU General Public License can be used, that proxy's
public statement of acceptance of a version permanently authorizes you
to choose that version for the Program.

  Later license versions may give you additional or different
permissions.  However, no additional obligations are imposed on any
author or copyright holder as a result of your choosing to follow a
later version.

  15. Disclaimer of Warranty.

  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. Limitation of Liability.

  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

  17. Interpretation of Sections 15 and 16.

  If the disclaimer of warranty and limitation of liability provided
above cannot be given local legal effect according to their terms,
reviewing courts shall apply local law that most closely approximates
an absolute waiver of all civil liability in connection with the
Program, unless a warranty or assumption of liability accompanies a
copy of the Program in return for a fee.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
state the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

  If the program does terminal interaction, make it output a short
notice like this when it starts in an interactive mode:

    <program>  Copyright (C) <year>  <name of author>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, your program's commands
might be different; for a GUI interface, you would use an "about box".

  You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU GPL, see
<http://www.gnu.org/licenses/>.

  The GNU General Public License does not permit incorporating your program
into proprietary programs.  If your program is a subroutine library, you
may consider it more useful to permit linking proprietary applications with
the library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.  But first, please read
<http://www.gnu.org/philosophy/why-not-lgpl.html>.
```

### LGPL-2.1+
```
                  GNU LESSER GENERAL PUBLIC LICENSE
                       Version 2.1, February 1999

 Copyright (C) 1991, 1999 Free Software Foundation, Inc.
 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

[This is the first released version of the Lesser GPL.  It also counts
 as the successor of the GNU Library Public License, version 2, hence
 the version number 2.1.]

                            Preamble

  The licenses for most software are designed to take away your
freedom to share and change it.  By contrast, the GNU General Public
Licenses are intended to guarantee your freedom to share and change
free software--to make sure the software is free for all its users.

  This license, the Lesser General Public License, applies to some
specially designated software packages--typically libraries--of the
Free Software Foundation and other authors who decide to use it.  You
can use it too, but we suggest you first think carefully about whether
this license or the ordinary General Public License is the better
strategy to use in any particular case, based on the explanations below.

  When we speak of free software, we are referring to freedom of use,
not price.  Our General Public Licenses are designed to make sure that
you have the freedom to distribute copies of free software (and charge
for this service if you wish); that you receive source code or can get
it if you want it; that you can change the software and use pieces of
it in new free programs; and that you are informed that you can do
these things.

  To protect your rights, we need to make restrictions that forbid
distributors to deny you these rights or to ask you to surrender these
rights.  These restrictions translate to certain responsibilities for
you if you distribute copies of the library or if you modify it.

  For example, if you distribute copies of the library, whether gratis
or for a fee, you must give the recipients all the rights that we gave
you.  You must make sure that they, too, receive or can get the source
code.  If you link other code with the library, you must provide
complete object files to the recipients, so that they can relink them
with the library after making changes to the library and recompiling
it.  And you must show them these terms so they know their rights.

  We protect your rights with a two-step method: (1) we copyright the
library, and (2) we offer you this license, which gives you legal
permission to copy, distribute and/or modify the library.

  To protect each distributor, we want to make it very clear that
there is no warranty for the free library.  Also, if the library is
modified by someone else and passed on, the recipients should know
that what they have is not the original version, so that the original
author's reputation will not be affected by problems that might be
introduced by others.

  Finally, software patents pose a constant threat to the existence of
any free program.  We wish to make sure that a company cannot
effectively restrict the users of a free program by obtaining a
restrictive license from a patent holder.  Therefore, we insist that
any patent license obtained for a version of the library must be
consistent with the full freedom of use specified in this license.

  Most GNU software, including some libraries, is covered by the
ordinary GNU General Public License.  This license, the GNU Lesser
General Public License, applies to certain designated libraries, and
is quite different from the ordinary General Public License.  We use
this license for certain libraries in order to permit linking those
libraries into non-free programs.

  When a program is linked with a library, whether statically or using
a shared library, the combination of the two is legally speaking a
combined work, a derivative of the original library.  The ordinary
General Public License therefore permits such linking only if the
entire combination fits its criteria of freedom.  The Lesser General
Public License permits more lax criteria for linking other code with
the library.

  We call this license the "Lesser" General Public License because it
does Less to protect the user's freedom than the ordinary General
Public License.  It also provides other free software developers Less
of an advantage over competing non-free programs.  These disadvantages
are the reason we use the ordinary General Public License for many
libraries.  However, the Lesser license provides advantages in certain
special circumstances.

  For example, on rare occasions, there may be a special need to
encourage the widest possible use of a certain library, so that it becomes
a de-facto standard.  To achieve this, non-free programs must be
allowed to use the library.  A more frequent case is that a free
library does the same job as widely used non-free libraries.  In this
case, there is little to gain by limiting the free library to free
software only, so we use the Lesser General Public License.

  In other cases, permission to use a particular library in non-free
programs enables a greater number of people to use a large body of
free software.  For example, permission to use the GNU C Library in
non-free programs enables many more people to use the whole GNU
operating system, as well as its variant, the GNU/Linux operating
system.

  Although the Lesser General Public License is Less protective of the
users' freedom, it does ensure that the user of a program that is
linked with the Library has the freedom and the wherewithal to run
that program using a modified version of the Library.

  The precise terms and conditions for copying, distribution and
modification follow.  Pay close attention to the difference between a
"work based on the library" and a "work that uses the library".  The
former contains code derived from the library, whereas the latter must
be combined with the library in order to run.

                  GNU LESSER GENERAL PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. This License Agreement applies to any software library or other
program which contains a notice placed by the copyright holder or
other authorized party saying it may be distributed under the terms of
this Lesser General Public License (also called "this License").
Each licensee is addressed as "you".

  A "library" means a collection of software functions and/or data
prepared so as to be conveniently linked with application programs
(which use some of those functions and data) to form executables.

  The "Library", below, refers to any such software library or work
which has been distributed under these terms.  A "work based on the
Library" means either the Library or any derivative work under
copyright law: that is to say, a work containing the Library or a
portion of it, either verbatim or with modifications and/or translated
straightforwardly into another language.  (Hereinafter, translation is
included without limitation in the term "modification".)

  "Source code" for a work means the preferred form of the work for
making modifications to it.  For a library, complete source code means
all the source code for all modules it contains, plus any associated
interface definition files, plus the scripts used to control compilation
and installation of the library.

  Activities other than copying, distribution and modification are not
covered by this License; they are outside its scope.  The act of
running a program using the Library is not restricted, and output from
such a program is covered only if its contents constitute a work based
on the Library (independent of the use of the Library in a tool for
writing it).  Whether that is true depends on what the Library does
and what the program that uses the Library does.

  1. You may copy and distribute verbatim copies of the Library's
complete source code as you receive it, in any medium, provided that
you conspicuously and appropriately publish on each copy an
appropriate copyright notice and disclaimer of warranty; keep intact
all the notices that refer to this License and to the absence of any
warranty; and distribute a copy of this License along with the
Library.

  You may charge a fee for the physical act of transferring a copy,
and you may at your option offer warranty protection in exchange for a
fee.

  2. You may modify your copy or copies of the Library or any portion
of it, thus forming a work based on the Library, and copy and
distribute such modifications or work under the terms of Section 1
above, provided that you also meet all of these conditions:

    a) The modified work must itself be a software library.

    b) You must cause the files modified to carry prominent notices
    stating that you changed the files and the date of any change.

    c) You must cause the whole of the work to be licensed at no
    charge to all third parties under the terms of this License.

    d) If a facility in the modified Library refers to a function or a
    table of data to be supplied by an application program that uses
    the facility, other than as an argument passed when the facility
    is invoked, then you must make a good faith effort to ensure that,
    in the event an application does not supply such function or
    table, the facility still operates, and performs whatever part of
    its purpose remains meaningful.

    (For example, a function in a library to compute square roots has
    a purpose that is entirely well-defined independent of the
    application.  Therefore, Subsection 2d requires that any
    application-supplied function or table used by this function must
    be optional: if the application does not supply it, the square
    root function must still compute square roots.)

These requirements apply to the modified work as a whole.  If
identifiable sections of that work are not derived from the Library,
and can be reasonably considered independent and separate works in
themselves, then this License, and its terms, do not apply to those
sections when you distribute them as separate works.  But when you
distribute the same sections as part of a whole which is a work based
on the Library, the distribution of the whole must be on the terms of
this License, whose permissions for other licensees extend to the
entire whole, and thus to each and every part regardless of who wrote
it.

Thus, it is not the intent of this section to claim rights or contest
your rights to work written entirely by you; rather, the intent is to
exercise the right to control the distribution of derivative or
collective works based on the Library.

In addition, mere aggregation of another work not based on the Library
with the Library (or with a work based on the Library) on a volume of
a storage or distribution medium does not bring the other work under
the scope of this License.

  3. You may opt to apply the terms of the ordinary GNU General Public
License instead of this License to a given copy of the Library.  To do
this, you must alter all the notices that refer to this License, so
that they refer to the ordinary GNU General Public License, version 2,
instead of to this License.  (If a newer version than version 2 of the
ordinary GNU General Public License has appeared, then you can specify
that version instead if you wish.)  Do not make any other change in
these notices.

  Once this change is made in a given copy, it is irreversible for
that copy, so the ordinary GNU General Public License applies to all
subsequent copies and derivative works made from that copy.

  This option is useful when you wish to copy part of the code of
the Library into a program that is not a library.

  4. You may copy and distribute the Library (or a portion or
derivative of it, under Section 2) in object code or executable form
under the terms of Sections 1 and 2 above provided that you accompany
it with the complete corresponding machine-readable source code, which
must be distributed under the terms of Sections 1 and 2 above on a
medium customarily used for software interchange.

  If distribution of object code is made by offering access to copy
from a designated place, then offering equivalent access to copy the
source code from the same place satisfies the requirement to
distribute the source code, even though third parties are not
compelled to copy the source along with the object code.

  5. A program that contains no derivative of any portion of the
Library, but is designed to work with the Library by being compiled or
linked with it, is called a "work that uses the Library".  Such a
work, in isolation, is not a derivative work of the Library, and
therefore falls outside the scope of this License.

  However, linking a "work that uses the Library" with the Library
creates an executable that is a derivative of the Library (because it
contains portions of the Library), rather than a "work that uses the
library".  The executable is therefore covered by this License.
Section 6 states terms for distribution of such executables.

  When a "work that uses the Library" uses material from a header file
that is part of the Library, the object code for the work may be a
derivative work of the Library even though the source code is not.
Whether this is true is especially significant if the work can be
linked without the Library, or if the work is itself a library.  The
threshold for this to be true is not precisely defined by law.

  If such an object file uses only numerical parameters, data
structure layouts and accessors, and small macros and small inline
functions (ten lines or less in length), then the use of the object
file is unrestricted, regardless of whether it is legally a derivative
work.  (Executables containing this object code plus portions of the
Library will still fall under Section 6.)

  Otherwise, if the work is a derivative of the Library, you may
distribute the object code for the work under the terms of Section 6.
Any executables containing that work also fall under Section 6,
whether or not they are linked directly with the Library itself.

  6. As an exception to the Sections above, you may also combine or
link a "work that uses the Library" with the Library to produce a
work containing portions of the Library, and distribute that work
under terms of your choice, provided that the terms permit
modification of the work for the customer's own use and reverse
engineering for debugging such modifications.

  You must give prominent notice with each copy of the work that the
Library is used in it and that the Library and its use are covered by
this License.  You must supply a copy of this License.  If the work
during execution displays copyright notices, you must include the
copyright notice for the Library among them, as well as a reference
directing the user to the copy of this License.  Also, you must do one
of these things:

    a) Accompany the work with the complete corresponding
    machine-readable source code for the Library including whatever
    changes were used in the work (which must be distributed under
    Sections 1 and 2 above); and, if the work is an executable linked
    with the Library, with the complete machine-readable "work that
    uses the Library", as object code and/or source code, so that the
    user can modify the Library and then relink to produce a modified
    executable containing the modified Library.  (It is understood
    that the user who changes the contents of definitions files in the
    Library will not necessarily be able to recompile the application
    to use the modified definitions.)

    b) Use a suitable shared library mechanism for linking with the
    Library.  A suitable mechanism is one that (1) uses at run time a
    copy of the library already present on the user's computer system,
    rather than copying library functions into the executable, and (2)
    will operate properly with a modified version of the library, if
    the user installs one, as long as the modified version is
    interface-compatible with the version that the work was made with.

    c) Accompany the work with a written offer, valid for at
    least three years, to give the same user the materials
    specified in Subsection 6a, above, for a charge no more
    than the cost of performing this distribution.

    d) If distribution of the work is made by offering access to copy
    from a designated place, offer equivalent access to copy the above
    specified materials from the same place.

    e) Verify that the user has already received a copy of these
    materials or that you have already sent this user a copy.

  For an executable, the required form of the "work that uses the
Library" must include any data and utility programs needed for
reproducing the executable from it.  However, as a special exception,
the materials to be distributed need not include anything that is
normally distributed (in either source or binary form) with the major
components (compiler, kernel, and so on) of the operating system on
which the executable runs, unless that component itself accompanies
the executable.

  It may happen that this requirement contradicts the license
restrictions of other proprietary libraries that do not normally
accompany the operating system.  Such a contradiction means you cannot
use both them and the Library together in an executable that you
distribute.

  7. You may place library facilities that are a work based on the
Library side-by-side in a single library together with other library
facilities not covered by this License, and distribute such a combined
library, provided that the separate distribution of the work based on
the Library and of the other library facilities is otherwise
permitted, and provided that you do these two things:

    a) Accompany the combined library with a copy of the same work
    based on the Library, uncombined with any other library
    facilities.  This must be distributed under the terms of the
    Sections above.

    b) Give prominent notice with the combined library of the fact
    that part of it is a work based on the Library, and explaining
    where to find the accompanying uncombined form of the same work.

  8. You may not copy, modify, sublicense, link with, or distribute
the Library except as expressly provided under this License.  Any
attempt otherwise to copy, modify, sublicense, link with, or
distribute the Library is void, and will automatically terminate your
rights under this License.  However, parties who have received copies,
or rights, from you under this License will not have their licenses
terminated so long as such parties remain in full compliance.

  9. You are not required to accept this License, since you have not
signed it.  However, nothing else grants you permission to modify or
distribute the Library or its derivative works.  These actions are
prohibited by law if you do not accept this License.  Therefore, by
modifying or distributing the Library (or any work based on the
Library), you indicate your acceptance of this License to do so, and
all its terms and conditions for copying, distributing or modifying
the Library or works based on it.

  10. Each time you redistribute the Library (or any work based on the
Library), the recipient automatically receives a license from the
original licensor to copy, distribute, link with or modify the Library
subject to these terms and conditions.  You may not impose any further
restrictions on the recipients' exercise of the rights granted herein.
You are not responsible for enforcing compliance by third parties with
this License.

  11. If, as a consequence of a court judgment or allegation of patent
infringement or for any other reason (not limited to patent issues),
conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot
distribute so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you
may not distribute the Library at all.  For example, if a patent
license would not permit royalty-free redistribution of the Library by
all those who receive copies directly or indirectly through you, then
the only way you could satisfy both it and this License would be to
refrain entirely from distribution of the Library.

If any portion of this section is held invalid or unenforceable under any
particular circumstance, the balance of the section is intended to apply,
and the section as a whole is intended to apply in other circumstances.

It is not the purpose of this section to induce you to infringe any
patents or other property right claims or to contest validity of any
such claims; this section has the sole purpose of protecting the
integrity of the free software distribution system which is
implemented by public license practices.  Many people have made
generous contributions to the wide range of software distributed
through that system in reliance on consistent application of that
system; it is up to the author/donor to decide if he or she is willing
to distribute software through any other system and a licensee cannot
impose that choice.

This section is intended to make thoroughly clear what is believed to
be a consequence of the rest of this License.

  12. If the distribution and/or use of the Library is restricted in
certain countries either by patents or by copyrighted interfaces, the
original copyright holder who places the Library under this License may add
an explicit geographical distribution limitation excluding those countries,
so that distribution is permitted only in or among countries not thus
excluded.  In such case, this License incorporates the limitation as if
written in the body of this License.

  13. The Free Software Foundation may publish revised and/or new
versions of the Lesser General Public License from time to time.
Such new versions will be similar in spirit to the present version,
but may differ in detail to address new problems or concerns.

Each version is given a distinguishing version number.  If the Library
specifies a version number of this License which applies to it and
"any later version", you have the option of following the terms and
conditions either of that version or of any later version published by
the Free Software Foundation.  If the Library does not specify a
license version number, you may choose any version ever published by
the Free Software Foundation.

  14. If you wish to incorporate parts of the Library into other free
programs whose distribution conditions are incompatible with these,
write to the author to ask for permission.  For software which is
copyrighted by the Free Software Foundation, write to the Free
Software Foundation; we sometimes make exceptions for this.  Our
decision will be guided by the two goals of preserving the free status
of all derivatives of our free software and of promoting the sharing
and reuse of software generally.

                            NO WARRANTY

  15. BECAUSE THE LIBRARY IS LICENSED FREE OF CHARGE, THERE IS NO
WARRANTY FOR THE LIBRARY, TO THE EXTENT PERMITTED BY APPLICABLE LAW.
EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR
OTHER PARTIES PROVIDE THE LIBRARY "AS IS" WITHOUT WARRANTY OF ANY
KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE
LIBRARY IS WITH YOU.  SHOULD THE LIBRARY PROVE DEFECTIVE, YOU ASSUME
THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN
WRITING WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MAY MODIFY
AND/OR REDISTRIBUTE THE LIBRARY AS PERMITTED ABOVE, BE LIABLE TO YOU
FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR
CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE
LIBRARY (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING
RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A
FAILURE OF THE LIBRARY TO OPERATE WITH ANY OTHER SOFTWARE), EVEN IF
SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
DAMAGES.

                     END OF TERMS AND CONDITIONS

           How to Apply These Terms to Your New Libraries

  If you develop a new library, and you want it to be of the greatest
possible use to the public, we recommend making it free software that
everyone can redistribute and change.  You can do so by permitting
redistribution under these terms (or, alternatively, under the terms of the
ordinary General Public License).

  To apply these terms, attach the following notices to the library.  It is
safest to attach them to the start of each source file to most effectively
convey the exclusion of warranty; and each file should have at least the
"copyright" line and a pointer to where the full notice is found.

    <one line to give the library's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Also add information on how to contact you by electronic and paper mail.

You should also get your employer (if you work as a programmer) or your
school, if any, to sign a "copyright disclaimer" for the library, if
necessary.  Here is a sample; alter the names:

  Yoyodyne, Inc., hereby disclaims all copyright interest in the
  library `Frob' (a library for tweaking knobs) written by James Random Hacker.

  <signature of Ty Coon>, 1 April 1990
  Ty Coon, President of Vice

That's all there is to it!
```

### MIT

```
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
```

### TGPPL

```
Transitive Grace Period Public Licence ("TGPPL") v. 1.0

This Transitive Grace Period Public Licence (the "License") applies to
any original work of authorship (the "Original Work") whose owner (the
"Licensor") has placed the following licensing notice adjacent to the
copyright notice for the Original Work:

*Licensed under the Transitive Grace Period Public Licence version 1.0*

   1. *Grant of Copyright License.* Licensor grants You a worldwide,
      royalty-free, non-exclusive, sublicensable license, for the
      duration of the copyright, to do the following:

         1. to reproduce the Original Work in copies, either alone or as
            part of a collective work;

         2. to translate, adapt, alter, transform, modify, or arrange
            the Original Work, thereby creating derivative works
            ("Derivative Works") based upon the Original Work;

         3. to distribute or communicate copies of the Original Work and
            Derivative Works to the public, with the proviso that copies
            of Original Work or Derivative Works that You distribute or
            communicate shall be licensed under this Transitive Grace
            Period Public Licence no later than 12 months after You
            distributed or communicated said copies;

         4. to perform the Original Work publicly; and

         5. to display the Original Work publicly.

   2. *Grant of Patent License.* Licensor grants You a worldwide,
      royalty-free, non-exclusive, sublicensable license, under patent
      claims owned or controlled by the Licensor that are embodied in
      the Original Work as furnished by the Licensor, for the duration
      of the patents, to make, use, sell, offer for sale, have made, and
      import the Original Work and Derivative Works.

   3. *Grant of Source Code License.* The term "Source Code" means the
      preferred form of the Original Work for making modifications to it
      and all available documentation describing how to modify the
      Original Work. Licensor agrees to provide a machine-readable copy
      of the Source Code of the Original Work along with each copy of
      the Original Work that Licensor distributes. Licensor reserves the
      right to satisfy this obligation by placing a machine-readable
      copy of the Source Code in an information repository reasonably
      calculated to permit inexpensive and convenient access by You for
      as long as Licensor continues to distribute the Original Work.

   4. *Exclusions From License Grant.* Neither the names of Licensor,
      nor the names of any contributors to the Original Work, nor any of
      their trademarks or service marks, may be used to endorse or
      promote products derived from this Original Work without express
      prior permission of the Licensor. Except as expressly stated
      herein, nothing in this License grants any license to Licensor's
      trademarks, copyrights, patents, trade secrets or any other
      intellectual property. No patent license is granted to make, use,
      sell, offer for sale, have made, or import embodiments of any
      patent claims other than the licensed claims defined in Section 2.
      No license is granted to the trademarks of Licensor even if such
      marks are included in the Original Work. Nothing in this License
      shall be interpreted to prohibit Licensor from licensing under
      terms different from this License any Original Work that Licensor
      otherwise would have a right to license.

   5. *External Deployment.* The term "External Deployment" means the
      use, distribution, or communication of the Original Work or
      Derivative Works in any way such that the Original Work or
      Derivative Works may be used by anyone other than You, whether
      those works are distributed or communicated to those persons or
      made available as an application intended for use over a network.
      As an express condition for the grants of license hereunder, You
      must treat any External Deployment by You of the Original Work or
      a Derivative Work as a distribution under section 1(c).

   6. *Attribution Rights.* You must retain, in the Source Code of any
      Derivative Works that You create, all copyright, patent, or
      trademark notices from the Source Code of the Original Work, as
      well as any notices of licensing and any descriptive text
      identified therein as an "Attribution Notice." You must cause the
      Source Code for any Derivative Works that You create to carry a
      prominent Attribution Notice reasonably calculated to inform
      recipients that You have modified the Original Work.

   7. *Warranty of Provenance and Disclaimer of Warranty.* Licensor
      warrants that the copyright in and to the Original Work and the
      patent rights granted herein by Licensor are owned by the Licensor
      or are sublicensed to You under the terms of this License with the
      permission of the contributor(s) of those copyrights and patent
      rights. Except as expressly stated in the immediately preceding
      sentence, the Original Work is provided under this License on an
      "AS IS" BASIS and WITHOUT WARRANTY, either express or implied,
      including, without limitation, the warranties of non-infringement,
      merchantability or fitness for a particular purpose. THE ENTIRE
      RISK AS TO THE QUALITY OF THE ORIGINAL WORK IS WITH YOU. This
      DISCLAIMER OF WARRANTY constitutes an essential part of this
      License. No license to the Original Work is granted by this
      License except under this disclaimer.

   8. *Limitation of Liability.* Under no circumstances and under no
      legal theory, whether in tort (including negligence), contract, or
      otherwise, shall the Licensor be liable to anyone for any
      indirect, special, incidental, or consequential damages of any
      character arising as a result of this License or the use of the
      Original Work including, without limitation, damages for loss of
      goodwill, work stoppage, computer failure or malfunction, or any
      and all other commercial damages or losses. This limitation of
      liability shall not apply to the extent applicable law prohibits
      such limitation.

   9. *Acceptance and Termination.* If, at any time, You expressly
      assented to this License, that assent indicates your clear and
      irrevocable acceptance of this License and all of its terms and
      conditions. If You distribute or communicate copies of the
      Original Work or a Derivative Work, You must make a reasonable
      effort under the circumstances to obtain the express assent of
      recipients to the terms of this License. This License conditions
      your rights to undertake the activities listed in Section 1,
      including your right to create Derivative Works based upon the
      Original Work, and doing so without honoring these terms and
      conditions is prohibited by copyright law and international
      treaty. Nothing in this License is intended to affect copyright
      exceptions and limitations (including 'fair use' or 'fair
      dealing'). This License shall terminate immediately and You may no
      longer exercise any of the rights granted to You by this License
      upon your failure to honor the conditions in Section 1(c).

  10. *Termination for Patent Action.* This License shall terminate
      automatically and You may no longer exercise any of the rights
      granted to You by this License as of the date You commence an
      action, including a cross-claim or counterclaim, against Licensor
      or any licensee alleging that the Original Work infringes a
      patent. This termination provision shall not apply for an action
      alleging patent infringement by combinations of the Original Work
      with other software or hardware.

  11. *Jurisdiction, Venue and Governing Law.* Any action or suit
      relating to this License may be brought only in the courts of a
      jurisdiction wherein the Licensor resides or in which Licensor
      conducts its primary business, and under the laws of that
      jurisdiction excluding its conflict-of-law provisions. The
      application of the United Nations Convention on Contracts for the
      International Sale of Goods is expressly excluded. Any use of the
      Original Work outside the scope of this License or after its
      termination shall be subject to the requirements and penalties of
      copyright or patent law in the appropriate jurisdiction. This
      section shall survive the termination of this License.

  12. *Attorneys' Fees.* In any action to enforce the terms of this
      License or seeking damages relating thereto, the prevailing party
      shall be entitled to recover its costs and expenses, including,
      without limitation, reasonable attorneys' fees and costs incurred
      in connection with such action, including any appeal of such
      action. This section shall survive the termination of this License.

  13. *Miscellaneous.* If any provision of this License is held to be
      unenforceable, such provision shall be reformed only to the extent
      necessary to make it enforceable.

  14. *Definition of "You" in This License.* "You" throughout this
      License, whether in upper or lower case, means an individual or a
      legal entity exercising rights under, and complying with all of
      the terms of, this License. For legal entities, "You" includes any
      entity that controls, is controlled by, or is under common control
      with you. For purposes of this definition, "control" means (i) the
      power, direct or indirect, to cause the direction or management of
      such entity, whether by contract or otherwise, or (ii) ownership
      of fifty percent (50%) or more of the outstanding shares, or (iii)
      beneficial ownership of such entity.

  15. *Right to Use.* You may use the Original Work in all ways not
      otherwise restricted or conditioned by this License or by law, and
      Licensor promises not to interfere with or be responsible for such
      uses by You.

  16. *Modification of This License.* This License is Copyright © 2007
      Zooko Wilcox-O'Hearn. Permission is granted to copy, distribute,
      or communicate this License without modification. Nothing in this
      License permits You to modify this License as applied to the
      Original Work or to Derivative Works. However, You may modify the
      text of this License and copy, distribute or communicate your
      modified version (the "Modified License") and apply it to other
      original works of authorship subject to the following conditions:
      (i) You may not indicate in any way that your Modified License is
      the "Transitive Grace Period Public Licence" or "TGPPL" and you
      may not use those names in the name of your Modified License; and
      (ii) You must replace the notice specified in the first paragraph
      above with the notice "Licensed under " or with a notice of your
      own that is not confusingly similar to the notice in this License.
```

### Zlib

```
This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
```
