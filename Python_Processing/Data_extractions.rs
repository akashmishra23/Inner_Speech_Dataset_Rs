use ndarray::{Array2, Axis};
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

fn extract_subject_from_bdf(root_dir: &str, n_s: i32, n_b: i32) -> Result<(Raw, String), Box<dyn Error>> {
    // Name correction if N_Subj is less than 10
    let num_s = sub_name(n_s);

    // Load data
    let file_name = format!(
        "{}/{}/ses-0{}/eeg/{}_ses-0{}_task-innerspeech_eeg.bdf",
        root_dir, num_s, n_b, num_s, n_b
    );

    let raw_data = mne::io::read_raw_bdf(input_fname: &file_name, preload: true, verbose: 'WARNING')?;

    Ok((raw_data, num_s))
}

fn extract_data_from_subject(root_dir: &str, n_s: i32, datatype: &str) -> Result<(Array2<f32>, Array2<i32>), Box<dyn Error>> {
    let mut data: Vec<Array2<f32>> = vec![];
    let mut y: Vec<Array2<i32>> = vec![];
    let n_b_arr = [1, 2, 3];
    let datatype = datatype.to_lowercase();

    for n_b in n_b_arr.iter() {
        // Name correction if N_Subj is less than 10
        let num_s = sub_name(n_s);

        let events = load_events(root_dir, n_s, n_b)?;

        let (file_name, data_tmp): (&str, Array2<f32>) = match datatype.as_str() {
            "eeg" => {
                let file_name = format!(
                    "{}/derivatives/{}/ses-0{}/{}_ses-0{}_eeg-epo.fif",
                    root_dir, num_s, n_b, num_s, n_b
                );
                let data_tmp = mne::read_epochs(&file_name, verbose: 'WARNING')?._data;
                (file_name.as_str(), data_tmp)
            }
            "exg" => {
                let file_name = format!(
                    "{}/derivatives/{}/ses-0{}/{}_ses-0{}_exg-epo.fif",
                    root_dir, num_s, n_b, num_s, n_b
                );
                let data_tmp = mne::read_epochs(&file_name, verbose: 'WARNING')?._data;
                (file_name.as_str(), data_tmp)
            }
            "baseline" => {
                let file_name = format!(
                    "{}/derivatives/{}/ses-0{}/{}_ses-0{}_baseline-epo.fif",
                    root_dir, num_s, n_b, num_s, n_b
                );
                let data_tmp = mne::read_epochs(&file_name, verbose: 'WARNING')?._data;
                (file_name.as_str(), data_tmp)
            }
            _ => return Err("Invalid Datatype".into()),
        };

        data.push(data_tmp);
        y.push(events);
    }

    let mut x_stacked = data[0].clone();
    let mut y_stacked = y[0].clone();

    for i in 1..data.len() {
        x_stacked = Array2::stack(Axis(0), &[&x_stacked.view(), &data[i].view()])?;
        y_stacked = Array2::stack(Axis(0), &[&y_stacked.view(), &y[i].view()])?;
    }

    Ok((x_stacked, y_stacked))
}

fn extract_block_data_from_subject(root_dir: &str, n_s: i32, datatype: &str, n_b: i32) -> Result<(Array2<f32>, Array2<i32>), Box<dyn Error>> {
    let num_s = sub_name(n_s);
    let events = load_events(root_dir, n_s, n_b)?;

    let (file_name, data): (&str, Array2<f32>) = match datatype.to_lowercase().as_str() {
        "eeg" => {
            let file_name = format!(
                "{}/derivatives/{}/ses-0{}/{}_ses-0{}_eeg-epo.fif",
                root_dir, num_s, n_b, num_s, n_b
            );
            let data = mne::read_epochs(&file_name, verbose: 'WARNING')?._data;
            (file_name.as_str(), data)
        }
        "exg" => {
            let file_name = format!(
                "{}/derivatives/{}/ses-0{}/{}_ses-0{}_exg-epo.fif",
                root_dir, num_s, n_b, num_s, n_b
            );
            let data = mne::read_epochs(&file_name, verbose: 'WARNING')?._data;
            (file_name.as_str(), data)
        }
        "baseline" => {
            let file_name = format!(
                "{}/derivatives/{}/ses-0{}/{}_ses-0{}_baseline-epo.fif",
                root_dir, num_s, n_b, num_s, n_b
            );
            let data = mne::read_epochs(&file_name, verbose: 'WARNING')?._data;
            (file_name.as_str(), data)
        }
        _ => return Err("Invalid Datatype".into()),
    };

    Ok((data, events))
}

fn extract_report(root_dir: &str, n_b: i32, n_s: i32) -> Result<Report, Box<dyn Error>> {
    let num_s = sub_name(n_s);
    let sub_dir = format!("{}/derivatives/{}/ses-0{}/{}_ses-0{}", root_dir, num_s, n_b, num_s, n_b);
    let file_name = format!("{}_report.pkl", sub_dir);
    let mut input_file = File::open(&file_name)?;
    let mut buffer = vec![];
    input_file.read_to_end(&mut buffer)?;

    let report: Report = bincode::deserialize(&buffer)?;

    Ok(report)
}

fn extract_tfr(trf_dir: &str, cond: &str, class_label: &str, tfr_method: &str, trf_type: &str) -> Result<TFR, Box<dyn Error>> {
    let (cond, class_label) = unify_names(cond, class_label);
    let file_name = format!("{}{}_{}_{}-tfr.h5", trf_dir, tfr_method, cond, class_label);
    let mut trf_file = File::open(&file_name)?;
    let mut buffer = vec![];
    trf_file.read_to_end(&mut buffer)?;

    let trf: TFR = bincode::deserialize(&buffer)?;

    Ok(trf)
}

fn extract_data_multisubject(root_dir: &str, n_s_list: &[i32], datatype: &str) -> Result<(Array2<f32>, Option<Array2<i32>>), Box<dyn Error>> {
    let n_b_arr = [1, 2, 3];
    let mut tmp_list_x: Vec
