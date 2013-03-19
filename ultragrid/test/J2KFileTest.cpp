// MoneyTest.cpp

#include <cppunit/config/SourcePrefix.h>
#include <unistd.h>
#include "J2KFileTest.h"
#include "image_hd.j2k.h"
#include "video_file/j2k.h"

using namespace std;

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( J2KFileTest );

J2KFileTest::J2KFileTest()
{
    char pattern[] = "/tmp/img.j2k.XXXXXX";
    m_name = string(mktemp(pattern));
    FILE *out = fopen(m_name.c_str(), "w");
    fwrite(imageBytes, sizeof(imageBytes), 1, out);
    fclose(out);
    m_name = "/var/tmp/j2k//00000000.j2k";
}

J2KFileTest::~J2KFileTest()
{
    unlink(m_name.c_str());
}

void
J2KFileTest::setUp()
{
}


void
J2KFileTest::tearDown()
{
}

void
J2KFileTest::testDesc()
{
    j2k_file test(m_name);
    struct video_desc desc = test.get_video_desc();

    // Check
    CPPUNIT_ASSERT_EQUAL(1920u, desc.width);
    CPPUNIT_ASSERT_EQUAL(1080u, desc.height);
}

struct UCharPtrDeleter
{
    void operator()(unsigned char *ptr) const
    {
        delete[] ptr;
    }
};

void
J2KFileTest::testData()
{
    j2k_file test(m_name);
    int len;
    std::shared_ptr<unsigned char> data(reinterpret_cast<unsigned char *>(test.get_raw_data(len)),
                    UCharPtrDeleter());

    CPPUNIT_ASSERT_EQUAL((int) sizeof(imageBytes), len);

    for(int i = 0; i < len; ++i) {
            if(data.get()[i] != imageBytes[i])
                    CPPUNIT_FAIL("Data returned doesn't match!");
    }
}

