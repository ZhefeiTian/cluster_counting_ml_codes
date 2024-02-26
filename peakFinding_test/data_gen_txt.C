#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TPolyMarker.h"
#include "TString.h"

#include <vector>
#include <iomanip>
#include <fstream>

using namespace std;

const int ndim = 1;
const int nleft = 5;
const int nright = 9;

extern "C" {

void find_bumps(vector<double> wf, vector<double>& time) {
    time.clear();
    for (int i = 1; i < wf.size() - 1; i++) {
        if (wf[i] - wf[i - 1] > wf[i+1] - wf[i]) {
            time.push_back(i);
        }
    }
}

bool local_max(vector<double> wf) {
    double amp_thr = 0.03;
    double delta_thr = 1e-3;

    if (wf[nleft] < amp_thr) return false;
    if (wf[nleft] - wf[nleft - 1] > delta_thr && wf[nleft] - wf[nleft + 1] > delta_thr) return true;

    return false;
}

void plot_wf(vector<double> wf, vector<double> time, TString label = "") {
    static int id = 0;

    TCanvas* c = new TCanvas("c", "c", 1200, 600);

    TString hname(Form("h_%d", id));
    int n = wf.size();
    TH1F* h = new TH1F(hname, hname, n, 0, n);
    for (int i = 0; i < n; i++) {
        h->SetBinContent(i+1, wf[i]);
    }
    h->SetTitle("");
    h->SetStats(0);
    h->Draw();
    //h->GetXaxis()->SetRange(350, 450);

    vector<double> amp(time.size());
    for (int i = 0; i < time.size(); i++) {
        amp[i] = wf[(int)time[i]];
    }
    TPolyMarker* m = new TPolyMarker(time.size(), &time[0], &amp[0]);
    m->SetMarkerStyle(20);
    m->SetMarkerColor(kRed);
    m->Draw("same");

    c->Print(Form("dataset_figs/wf_%d%s.png", id, label.Data()));
    delete c;

    id++;
}

void plot_sample(vector<double> sample, TString label = "") {
    static int id = 0;

    TCanvas* c = new TCanvas("c", "c", 800, 600);

    TString hname(Form("data_%d", id));
    int n = sample.size();
    TH1F* h = new TH1F(hname, hname, n, 0, n);
    for (int i = 0; i < n; i++) {
        h->SetBinContent(i+1, sample[i]);
    }
    h->SetTitle("");
    h->SetStats(0);
    h->SetLineWidth(2);
    h->Draw();
    //h->GetXaxis()->SetRange(350, 450);

    c->Print(Form("dataset_figs/sample_%d%s.png", id, label.Data()));
    delete c;

    id++;
}

void data_gen_txt(TString filename = "signal_noise05_500k_2.root", TString outfile = "dataset_for_test.txt", int start = 0, int n = -1, int type = 0, bool isData = false) { // tag 0: supervised, tag 1: only signal, tag 2: only background
    int ntime = nleft + 1 + nright;

    TFile* f = new TFile(filename);
    TTree* t = (TTree*)f->Get("sim");
    vector<double>* wf = 0;
    vector<double>* time = 0;
    vector<double>* tag = 0;
    t->SetBranchAddress("wf_i", &wf);
    if (!isData) t->SetBranchAddress("time", &time);
    if (!isData) t->SetBranchAddress("tag", &tag);
    if (isData) {
        time = new vector<double>(0);
    }
    int evtno, id;
    double peak_time;
    double shift, sigma;
    ofstream output(outfile);

    // Header
    output << setw(8) << "EvtNo" << ", ";
    output << setw(4) << "ID" << ", ";
    output << setw(8) << "Shift" << ", ";
    output << setw(8) << "Sigma" << ", ";
    output << setw(8) << "Time" << ", ";
    for (int itime = 0; itime < ntime; itime++)
    {
        output << setw(8) << Form("Time%d", itime);
        if (itime != ntime - 1)
            output << ", ";
        else
            output << endl;
    }

    // Data
    /*vector<double> sample;
    int ntot = t->GetEntries();
    int nsample = 1000 > ntot ? ntot : 1000;
    for (int i = 0; i < nsample; i++) {
        t->GetEntry(i);

        for (int j = 0; j < time->size(); j++) {
            double tpeak = (*time)[j];
            if (tpeak < nleft || tpeak > wf->size() - nright - 1) continue;

            int ipeak = (int)tpeak;
            double avg = 0.;
            for (int itime = 0; itime < ntime; itime++) {
                double val = (*wf)[ipeak - nleft + itime];
                avg += val;
            }
            for (int itime = 0; itime < ntime; itime++) {
                sample.push_back((*wf)[ipeak - nleft + itime] - avg/ntime);
            }
        }
    }

    double rms = 0.;
    for (int i = 0; i < sample.size(); i++) {
        rms += sample[i] * sample[i];
    }
    rms = sqrt(rms/sample.size());*/

    int ntot = t->GetEntries();
    int end;
    if (n > 0) end = start + n > ntot ? ntot : start + n;
    else end = ntot;
    int ndata = 0;
    TString label;
    if (type == 1) label = "_sig";
    if (type == 2) label = "_bkg";
    for (int i = start; i < end; i++) {
        if (i % 1000 == 0) cout << "Processing event " << i << " ..." << endl;
        t->GetEntry(i);

        evtno = i;
        vector<double> selected_times;

        if (isData) {
            for (int i = 0; i < wf->size(); i++) {
                (*wf)[i] *= -1.;
            }
            find_bumps(*wf, *time);
        }
// OLD: 0=pri, 1=sec, 2=bkg
// NEW: 0=bkg, 1=pri, 2=sec
        for (int j = 0; j < time->size(); j++) {
            if (type == 0) {
                if (!isData) {
                    if ((*tag)[j] > 0) id = 1;
                    else id = 0;
                }
                else {
                    id = -1;
                }
            }
            if (type == 1) {
                id = 1;
            }
            if (type == 2) id = 0;

            double tpeak = (*time)[j];
            if (tpeak < nleft || tpeak > wf->size() - nright - 1) continue;

            int ipeak = (int)tpeak;
            double avg = 0.;
            vector<double> wf_slice(ntime);
            for (int itime = 0; itime < ntime; itime++) {
                double val = (*wf)[ipeak - nleft + itime];
                wf_slice[itime] = val;
                avg += val;
            }

            if (type == 1 && !local_max(wf_slice)) continue;
            selected_times.push_back(tpeak);

            for (int itime = 0; itime < ntime; itime++) {
                wf_slice[itime] -= avg/ntime;
                wf_slice[itime] /= 1.;
            }
            peak_time = tpeak;
            shift = avg/ntime;
            sigma = 1.;

            // if (ndata < 10) plot_sample(wf_slice, label);

            output << setw(8) << evtno << ", ";
            output << setw(4) << id << ", ";
            output << setw(8) << setprecision(4) << shift << ", ";
            output << setw(8) << setprecision(4) << sigma << ", ";
            output << setw(8) << setprecision(4) << peak_time << ", ";
            for (int itime = 0; itime < ntime; itime++) {
                output << setw(8) << setprecision(4) << wf_slice[itime];
                if (itime != ntime - 1) output << ", ";
                else output << endl;
            }

            ndata++;
        }

        // if (i < 10) plot_wf(*wf, selected_times, label);
    }

    //f->Close();
    output.close();
}

} // extern "C"
