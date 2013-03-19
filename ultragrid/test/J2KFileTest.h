#ifndef J2KFILETEST_H
#define J2KFILETEST_H

#include <cppunit/extensions/HelperMacros.h>
#include <string>

class J2KFileTest : public CPPUNIT_NS::TestFixture
{
  CPPUNIT_TEST_SUITE( J2KFileTest );
  CPPUNIT_TEST( testDesc );
  CPPUNIT_TEST( testData );
  CPPUNIT_TEST_SUITE_END();

public:
  J2KFileTest();
  ~J2KFileTest();
  void setUp();
  void tearDown();

  void testDesc();
  void testData();
private:
  std::string m_name; // test J2K file name
};

#endif //  J2KFILETEST_H
