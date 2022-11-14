#include <string>
#include <vector>
#include "grainingwindow.h"
#include <QApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include "utils/filters.cuh"

int32 main(int32 argc,char **argv)
{
  QApplication app(argc, argv);
  QCoreApplication::setOrganizationName("Damien Boildieu - XLim");
  QCoreApplication::setApplicationName("Graining");
  QCoreApplication::setApplicationVersion(QT_VERSION_STR);
  QCommandLineParser parser;
  parser.setApplicationDescription(QCoreApplication::applicationName());
  parser.addHelpOption();
  parser.addVersionOption();
  QCommandLineOption fileOption(QStringList() << "f" << "file", "The file to grain.", "file");
  parser.addOption(fileOption);
  parser.process(app);

  GrainingWindow window;
  if (parser.isSet(fileOption)) {
    window.setImagePath(parser.value(fileOption));
    window.loadImage();
  }
  window.show();
  return app.exec();
}

