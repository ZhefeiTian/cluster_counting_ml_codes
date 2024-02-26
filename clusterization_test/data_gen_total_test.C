int n_size = 3000;

void data_gen_total_test(TString infile = "counting_test_15.root", TString outfile = "peaks_mean15.txt", TString outfile2 = "times_mean15.txt", int start = 0, int n = -1) {
    TFile* f = new TFile(infile);
    TTree* t = (TTree*)f->Get("sim");

    double sampling_rate;
    std::vector<double>* time = 0;
    std::vector<int>* id = 0;
    t->SetBranchAddress("sampling_rate", &sampling_rate);
    t->SetBranchAddress("count_x", &time);
    t->SetBranchAddress("id", &id);

    ofstream output(outfile);
    ofstream output2(outfile2);

    int nentries = t->GetEntries();
    int end = start + n > nentries ? nentries : start + n;
    if (n < 0) end = nentries;
    for (int i = start; i < end; i++) {
        if (i % 1000 == 0) cout << "Processing event " << i << " ..." << endl;
        t->GetEntry(i);

        //cout << "For evt" << i << endl;

        std::vector<int> peaks;
        std::vector<int> times;
        int npri = 0;
        for (int j = 0; j < time->size(); j++) {
            int idx = (*time)[j]/sampling_rate;
            if (idx ==0 || idx >= n_size) continue;
            //tvec[idx] = 1;
            if ((*id)[j] == 0) {
                peaks.push_back(0);
                times.push_back(idx);
            }
            if ((*id)[j] == 1) {
                peaks.push_back(1);
                times.push_back(idx);
            }
            if ((*id)[j] == 2) {
                peaks.push_back(2);
                times.push_back(idx);
            }
            // cout << setw(9) << (*time)[j];
            // cout << setw(9) << (*time)[j]/sampling_rate;
            // cout << setw(5) << typeid((*time)[j]/sampling_rate).name();
            // cout << setw(5) << idx;
            // cout << setw(2) << (*id)[j];
            // cout << setw(4) << j+1 << "/" << time->size() << endl;
        }
        int peaks_size = peaks.size();
        for (int j = 0; j < peaks_size; j++) {
            output << setw(5) << peaks[j];
        }
        output << endl;
        for (int j = 0; j < peaks_size; j++) {
            output2 << setw(5) << times[j];
        }
        output2 << endl;
    }

    output.close();
    output2.close();
}
