// kerf_viewer.cpp — Qt6 Widgets GUI for interactive dicing-kerf inspection.
//
// Loads a PGM scribe-line image, runs the wproc KerfInspector with parameters
// editable in the side panel, and shows the annotated overlay, the pass/fail
// verdict, and a per-chip table. Results can be saved as PNG + JSON.
//
// Headless self-test (used in CI / containers without a display):
//   QT_QPA_PLATFORM=offscreen kerf_viewer --selftest shot.png sample.pgm
// renders the full window off-screen, saves a screenshot, prints the JSON
// report, and exits with the inspection verdict (0 = pass, 1 = fail).
#include <QApplication>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QMessageBox>
#include <QPixmap>
#include <QPushButton>
#include <QScrollArea>
#include <QStatusBar>
#include <QTableWidget>
#include <QVBoxLayout>

#include <fstream>
#include <iostream>
#include <string>

#include "wproc/io.hpp"
#include "wproc/kerf_inspect.hpp"

namespace {

QImage to_qimage(const wproc::ColorImage& img) {
    QImage out(img.width(), img.height(), QImage::Format_RGB888);
    const int row_bytes = img.width() * 3;
    for (int y = 0; y < img.height(); ++y)
        std::copy_n(img.data() + static_cast<std::size_t>(y) * row_bytes,
                    row_bytes, out.scanLine(y));
    return out;
}

}  // namespace

class ViewerWindow : public QMainWindow {
public:
    ViewerWindow() {
        setWindowTitle("wproc kerf viewer — dicing street inspection");
        resize(1100, 560);

        auto* central = new QWidget(this);
        auto* rootLayout = new QHBoxLayout(central);

        // Left: annotated image in a scroll area (2x zoom for visibility).
        imageLabel_ = new QLabel(tr("Open a PGM scribe-line image to begin."));
        imageLabel_->setAlignment(Qt::AlignCenter);
        auto* scroll = new QScrollArea;
        scroll->setWidget(imageLabel_);
        scroll->setWidgetResizable(true);
        rootLayout->addWidget(scroll, /*stretch=*/1);

        // Right: verdict, parameters, chip table, actions.
        auto* side = new QVBoxLayout;

        verdict_ = new QLabel(tr("—"));
        verdict_->setAlignment(Qt::AlignCenter);
        verdict_->setFixedHeight(44);
        setVerdictStyle("#666", tr("NO IMAGE"));
        side->addWidget(verdict_);

        summary_ = new QLabel(tr("width: —\nconfidence: —"));
        side->addWidget(summary_);

        auto* paramBox = new QGroupBox(tr("Inspection parameters"));
        auto* form = new QFormLayout(paramBox);
        umPerPx_ = makeSpin(0.01, 100.0, 0.5, 3);
        nominal_ = makeSpin(0.0, 10000.0, 40.0, 1);
        tol_ = makeSpin(-1.0, 10000.0, 5.0, 1);
        maxChip_ = makeSpin(0.0, 10000.0, 5.0, 2);
        blur_ = makeSpin(0.1, 50.0, 1.2, 2);
        form->addRow(tr("µm per pixel"), umPerPx_);
        form->addRow(tr("nominal width µm"), nominal_);
        form->addRow(tr("width tol ± µm"), tol_);
        form->addRow(tr("max chip µm"), maxChip_);
        form->addRow(tr("blur sigma px"), blur_);
        side->addWidget(paramBox);

        chips_ = new QTableWidget(0, 4);
        chips_->setHorizontalHeaderLabels(
            {tr("chip"), tr("side"), tr("protrusion µm"), tr("area µm²")});
        chips_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        chips_->setEditTriggers(QAbstractItemView::NoEditTriggers);
        side->addWidget(chips_, /*stretch=*/1);

        auto* openBtn = new QPushButton(tr("Open PGM…"));
        auto* runBtn = new QPushButton(tr("Re-inspect"));
        auto* saveBtn = new QPushButton(tr("Save annotated PNG…"));
        auto* jsonBtn = new QPushButton(tr("Save JSON report…"));
        side->addWidget(openBtn);
        side->addWidget(runBtn);
        side->addWidget(saveBtn);
        side->addWidget(jsonBtn);

        auto* sideWidget = new QWidget;
        sideWidget->setLayout(side);
        sideWidget->setFixedWidth(320);
        rootLayout->addWidget(sideWidget);
        setCentralWidget(central);

        connect(openBtn, &QPushButton::clicked, this, [this] {
            QString path = QFileDialog::getOpenFileName(
                this, tr("Open scribe-line image"), {}, tr("PGM images (*.pgm)"));
            if (!path.isEmpty()) loadAndInspect(path.toStdString());
        });
        connect(runBtn, &QPushButton::clicked, this, [this] { inspect(); });
        connect(saveBtn, &QPushButton::clicked, this, [this] {
            if (annotated_.isNull()) return;
            QString path = QFileDialog::getSaveFileName(
                this, tr("Save annotated image"), "annotated.png",
                tr("PNG (*.png)"));
            if (!path.isEmpty()) annotated_.save(path);
        });
        connect(jsonBtn, &QPushButton::clicked, this, [this] {
            if (!hasImage_) return;
            QString path = QFileDialog::getSaveFileName(
                this, tr("Save JSON report"), "report.json", tr("JSON (*.json)"));
            if (path.isEmpty()) return;
            std::ofstream out(path.toStdString());
            out << wproc::to_json(result_);
        });
    }

    bool loadAndInspect(const std::string& path) {
        try {
            image_ = wproc::read_pgm(path);
        } catch (const std::exception& e) {
            QMessageBox::warning(this, tr("Load failed"), e.what());
            return false;
        }
        hasImage_ = true;
        statusBar()->showMessage(QString::fromStdString(path));
        inspect();
        return true;
    }

    void inspect() {
        if (!hasImage_) return;
        wproc::KerfInspector::Params p;
        p.um_per_px = umPerPx_->value();
        p.nominal_width_um = nominal_->value();
        p.width_tol_um = tol_->value();
        p.max_allowed_chip_um = maxChip_->value();
        p.blur_sigma = static_cast<float>(blur_->value());
        wproc::KerfInspector inspector(p);
        result_ = inspector.inspect(image_);

        annotated_ = to_qimage(inspector.annotate(image_, result_));
        imageLabel_->setPixmap(QPixmap::fromImage(annotated_).scaled(
            annotated_.width() * 2, annotated_.height() * 2,
            Qt::KeepAspectRatio, Qt::FastTransformation));
        imageLabel_->adjustSize();

        if (!result_.found) {
            setVerdictStyle("#8a6d00", tr("NO KERF FOUND"));
        } else if (result_.pass) {
            setVerdictStyle("#1a9e57", tr("PASS"));
        } else {
            setVerdictStyle("#d23b3b", tr("FAIL"));
        }
        summary_->setText(
            tr("width: %1 µm  (spec %2 ± %3)\nmax chip: %4 µm   confidence: %5")
                .arg(result_.width_um, 0, 'f', 2)
                .arg(nominal_->value(), 0, 'f', 1)
                .arg(tol_->value(), 0, 'f', 1)
                .arg(result_.max_chip_um, 0, 'f', 2)
                .arg(result_.confidence, 0, 'f', 2));

        chips_->setRowCount(static_cast<int>(result_.chips.size()));
        for (int r = 0; r < static_cast<int>(result_.chips.size()); ++r) {
            const auto& c = result_.chips[static_cast<std::size_t>(r)];
            chips_->setItem(r, 0, new QTableWidgetItem(QString::number(c.id)));
            chips_->setItem(r, 1,
                            new QTableWidgetItem(c.side < 0 ? tr("left")
                                                            : tr("right")));
            chips_->setItem(
                r, 2, new QTableWidgetItem(QString::number(c.protrusion_um, 'f', 2)));
            chips_->setItem(
                r, 3, new QTableWidgetItem(QString::number(c.area_um2, 'f', 1)));
        }
    }

    const wproc::KerfResult& result() const { return result_; }

private:
    QDoubleSpinBox* makeSpin(double lo, double hi, double val, int decimals) {
        auto* s = new QDoubleSpinBox;
        s->setRange(lo, hi);
        s->setDecimals(decimals);
        s->setValue(val);
        return s;
    }
    void setVerdictStyle(const char* color, const QString& text) {
        verdict_->setText(text);
        verdict_->setStyleSheet(
            QString("background:%1;color:white;font-weight:bold;font-size:18px;"
                    "border-radius:6px;")
                .arg(color));
    }

    QLabel* imageLabel_ = nullptr;
    QLabel* verdict_ = nullptr;
    QLabel* summary_ = nullptr;
    QTableWidget* chips_ = nullptr;
    QDoubleSpinBox *umPerPx_ = nullptr, *nominal_ = nullptr, *tol_ = nullptr,
                   *maxChip_ = nullptr, *blur_ = nullptr;

    wproc::Image image_;
    bool hasImage_ = false;
    wproc::KerfResult result_;
    QImage annotated_;
};

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QStringList args = app.arguments();

    // Headless self-test: --selftest <shot.png> <input.pgm>
    int selftestIdx = args.indexOf("--selftest");
    if (selftestIdx >= 0) {
        if (selftestIdx + 2 >= args.size()) {
            std::cerr << "usage: kerf_viewer --selftest <shot.png> <input.pgm>\n";
            return 2;
        }
        ViewerWindow w;
        if (!w.loadAndInspect(args[selftestIdx + 2].toStdString())) return 3;
        w.show();
        w.grab().save(args[selftestIdx + 1]);
        std::cout << wproc::to_json(w.result());
        return w.result().pass ? 0 : 1;
    }

    ViewerWindow w;
    w.show();
    if (args.size() > 1) w.loadAndInspect(args[1].toStdString());
    return app.exec();
}
