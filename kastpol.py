#!/usr/bin/env python
# coding: utf-8
# Sergiy S. Vasylyev, Kishore Patra
import glob
import pickle
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import ascii
from scipy import signal
from astropy.table import Table 
from matplotlib.ticker import MaxNLocator

try:
    import spolpat3 as sp
except ImportError:
    print("ERROR: Could not import spolpat3.py.")
    print("Please check if spolpat3.py is in the same directory as this script,")
    print("or that it's in your Python path.")
    exit()

# --- Helper Functions for User Input ---
def get_yes_no(prompt):
    """Gets a yes/no answer from the user."""
    while True:
        response = input(prompt + " (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

def get_int(prompt, default=None):
    """Gets an integer from the user."""
    while True:
        prompt_full = prompt
        if default is not None:
            prompt_full += f" (default: {default})"
        response = input(prompt_full + ": ").strip()
        if not response and default is not None:
            return default
        try:
            return int(response)
        except ValueError:
            print("Invalid input. Please enter an integer.")

def get_float(prompt, default=None):
    """Gets a float from the user."""
    while True:
        prompt_full = prompt
        if default is not None:
            prompt_full += f" (default: {default})"
        response = input(prompt_full + ": ").strip()
        if not response and default is not None:
            return default
        try:
            return float(response)
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_string(prompt, default=None, allow_empty=False):
    """Gets a string from the user."""
    while True:
        prompt_full = prompt
        if default is not None:
            prompt_full += f" (default: '{default}')"
        response = input(prompt_full + ": ").strip()
        if not response and default is not None:
            return default
        if response or allow_empty:
            return response
        print("This field cannot be empty.")

# --- read_trace_files Function (with Diagnostic Prints) ---
def read_trace_files(base_name, loop_suffix_from_input=""):
    traces = []
    angles = ["0.0", "22.5", "45.0", "67.5"]
    file_data = {"bottom": [], "top": []}
    all_files_found_and_read = True
    descriptive_name = base_name + loop_suffix_from_input
    print(f"\n--- Reading files for: {descriptive_name} ---")

    if not base_name:
        print("Error in read_trace_files: base_name is empty.")
        return None

    for dir_prefix in ["bottom", "top"]:
        print(f"\n# Searching in directory: '{dir_prefix}/'")
        for angle_str in angles:
            glob_pattern = f"{dir_prefix}/{base_name}{angle_str}{loop_suffix_from_input}*.flm"
            found_files = glob.glob(glob_pattern)
            
            print(f"  - Waveplate Angle: {angle_str}")
            print(f"    - Searching with pattern: {glob_pattern}")
            
            if found_files:
                filename_to_read = found_files[0]
                print(f"    - Found {len(found_files)} file(s). Using: '{filename_to_read}'")
                if len(found_files) > 1:
                    print(f"    - WARNING: Found multiple files for this angle. Only the first one was used.")
                try:
                    file_data[dir_prefix].append(ascii.read(filename_to_read))
                except Exception as e:
                    print(f"    - ERROR reading file '{filename_to_read}': {e}")
                    all_files_found_and_read = False
                    break
            else:
                print(f"    - ERROR: File not found for this angle.")
                all_files_found_and_read = False
                break
        if not all_files_found_and_read:
            break
    
    if not all_files_found_and_read:
        print(f"\nFailed to read all required files for {descriptive_name}.")
        return None

    if len(file_data["bottom"]) == 4 and len(file_data["top"]) == 4:
        traces.extend(file_data["bottom"])
        traces.extend(file_data["top"])
        print(f"\nSuccessfully read 8 files for {descriptive_name}.")
        return traces
    else:
        print(f"\nError: Did not collect 8 trace files for {descriptive_name}. Bottom: {len(file_data['bottom'])}, Top: {len(file_data['top'])}")
        return None

# --- Helper functions from Q_U_plots.py (for integration) ---
def qu_rebin_data(list_to_rebin, binning_val, n_old_val):
    if not isinstance(list_to_rebin, np.ndarray):
        list_to_rebin = np.array(list_to_rebin)
    if list_to_rebin.size == 0: return np.array([])
    if n_old_val <= 0: return list_to_rebin
    
    step = int(round(binning_val / n_old_val))
    if step <= 0: step = 1

    new_list = []
    if list_to_rebin.size < step:
        return np.array([np.median(list_to_rebin)]) if list_to_rebin.size > 0 else np.array([])

    for i in range(0, list_to_rebin.size - step + 1, step):
        new_list.append(np.median(list_to_rebin[i:i + step]))
    return np.array(new_list)

def qu_rebin_err_data(err_array, binning_val, n_old_val):
    if not isinstance(err_array, np.ndarray): 
        err_array = np.array(err_array)
    if err_array.size == 0: return np.array([])
    if n_old_val <= 0 or binning_val <= 0: return err_array
    
    squared_errors = err_array**2
    rebinned_squared_errors = qu_rebin_data(squared_errors, binning_val, n_old_val)
    
    if rebinned_squared_errors.size == 0: return np.array([])

    factor = n_old_val / binning_val
    if factor <= 0 : return err_array 
    
    final_err = np.sqrt(factor * rebinned_squared_errors)
    return final_err

# --- New function for Q/U Plotting and Export (based on QU_plots.py ) ---
def run_qu_plotting_and_export(
    source_data_tuple, 
    obj_name_base_for_qu, 
    n_old_stokes_for_qu, 
    main_script_flipq_setting 
    ):
    print("\n" + "=" * 50)
    print("--- Q/U Plotting and Final Data Export ---")

    q_u_obj_name = get_string("Enter Object Name for Q/U plots & final .dat filenames (e.g., SN2024iss)", default=obj_name_base_for_qu)
    q_u_epoch_name = get_string("Enter Epoch Name for Q/U plots & final .dat filenames (e.g., akm)", default="final_obs")
    q_u_redshift = get_float("Enter object redshift for Q/U plots (e.g., 0.0008)", default=0.0008)
    
    default_qu_binning = 30 
    if n_old_stokes_for_qu > 0 : default_qu_binning = int(15 * n_old_stokes_for_qu) 
    if default_qu_binning < n_old_stokes_for_qu : default_qu_binning = int(n_old_stokes_for_qu) 
    
    q_u_wave_binning = get_int(f"Enter wavelength binning for final Q/U/P/PA data & plots (Angstroms, e.g., 18)", default=max(int(n_old_stokes_for_qu), default_qu_binning))
    q_u_flux_spec_bin_display = get_int("Spectral binning for *flux display only* in Q/U plot (Angstroms, e.g., 2)", default=max(int(n_old_stokes_for_qu // 2), 2))

    s_wave, s_q, s_u, s_q_err, s_u_err, s_p_frac, s_p_err_frac, s_pa_rad, s_pa_err_rad, s_flux, s_flux_err = source_data_tuple
    
    qu_plot_labels = [f"{q_u_obj_name} {q_u_epoch_name}"]

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize=(10, 10))
    plt.subplots_adjust(wspace=0., hspace=0.)
    plt.minorticks_on()
    ax1.tick_params(labelbottom=False, labeltop=True)
    ax5.set_xlabel(f"Rest Wavelength [$\AA$] (z={q_u_redshift:.4f})", fontweight="bold")
    ax1.set_ylabel("Scaled Flux", fontweight="bold")
    ax2.set_ylabel("q [%]", fontweight="bold")
    ax3.set_ylabel("u [%]", fontweight="bold")
    ax4.set_ylabel("p [%]", fontweight="bold")
    ax5.set_ylabel("PA [deg]", fontweight="bold")
    for ax_pol_tick in [ax2,ax3,ax4,ax5]: ax_pol_tick.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
    ax2.set_ylim(-2.5, 2.5); ax3.set_ylim(-2.5, 2.5); ax4.set_ylim(-0.5, 3.5); ax5.set_ylim(0, 360) 

    wave_proc, Q_proc, U_proc, Q_err_proc, U_err_proc, flux_proc, flux_err_proc = s_wave, s_q, s_u, s_q_err, s_u_err, s_flux, s_flux_err
    
    n_old_A_eff_qu = np.median(np.diff(wave_proc)) if len(wave_proc) > 1 else n_old_stokes_for_qu
    if n_old_A_eff_qu <=0: n_old_A_eff_qu = n_old_stokes_for_qu 

    wave_rest_proc = wave_proc / (1. + q_u_redshift)
    n_old_A_rest_eff_qu = n_old_A_eff_qu / (1. + q_u_redshift)
    if n_old_A_rest_eff_qu <=0: n_old_A_rest_eff_qu = 1.0 

    if len(wave_rest_proc) > 0 and q_u_flux_spec_bin_display > 0 and n_old_A_rest_eff_qu > 0:
        flux_wave_plot = qu_rebin_data(wave_rest_proc, q_u_flux_spec_bin_display, n_old_A_rest_eff_qu)
        flux_val_plot = qu_rebin_data(flux_proc, q_u_flux_spec_bin_display, n_old_A_rest_eff_qu)
        flux_err_plot = qu_rebin_err_data(flux_err_proc, q_u_flux_spec_bin_display, n_old_A_rest_eff_qu)
        if len(flux_wave_plot) > 0 and len(flux_val_plot) > 0:
            med_flux = np.nanmedian(flux_val_plot)
            if med_flux != 0 and np.isfinite(med_flux):
                flux_val_plot /= med_flux
                if len(flux_err_plot) == len(flux_val_plot): flux_err_plot /= med_flux
            ax1.plot(flux_wave_plot, flux_val_plot, color='k', label=qu_plot_labels[0], lw=1.0)
            if len(flux_err_plot) == len(flux_val_plot):
                 ax1.fill_between(flux_wave_plot, flux_val_plot - flux_err_plot, flux_val_plot + flux_err_plot, color='k', alpha=0.2)
    ax1.set_title(f"{q_u_obj_name} - {q_u_epoch_name}", y=1.0, pad=-14, loc="right", fontweight="bold", fontsize=10)
    if ax1.has_data(): ax1.legend(loc='upper left', fontsize=8)

    wave_final_qu, Q_final_qu, U_final_qu, Q_err_final_qu, U_err_final_qu, flux_for_dat_qu, flux_err_for_dat_qu = \
        wave_rest_proc, Q_proc, U_proc, Q_err_proc, U_err_proc, flux_proc, flux_err_proc

    if q_u_wave_binning > 0 and n_old_A_rest_eff_qu > 0 and len(wave_rest_proc) > 0:
        binned_wave_check = qu_rebin_data(wave_rest_proc, q_u_wave_binning, n_old_A_rest_eff_qu)
        if len(binned_wave_check) > 0:
            wave_final_qu = binned_wave_check
            Q_final_qu = qu_rebin_data(Q_proc, q_u_wave_binning, n_old_A_rest_eff_qu)
            Q_err_final_qu = qu_rebin_err_data(Q_err_proc, q_u_wave_binning, n_old_A_rest_eff_qu)
            U_final_qu = qu_rebin_data(U_proc, q_u_wave_binning, n_old_A_rest_eff_qu)
            U_err_final_qu = qu_rebin_err_data(U_err_proc, q_u_wave_binning, n_old_A_rest_eff_qu)
            flux_for_dat_qu = qu_rebin_data(flux_proc, q_u_wave_binning, n_old_A_rest_eff_qu)
            flux_err_for_dat_qu = qu_rebin_err_data(flux_err_proc, q_u_wave_binning, n_old_A_rest_eff_qu)
    
    p_biased = np.sqrt(Q_final_qu**2 + U_final_qu**2)
    p_err = np.zeros_like(p_biased)
    mask_p_gt_0 = p_biased > 1e-9
    p_err[mask_p_gt_0] = (1. / p_biased[mask_p_gt_0]) * np.sqrt(
        (Q_final_qu[mask_p_gt_0] * Q_err_final_qu[mask_p_gt_0])**2 + (U_final_qu[mask_p_gt_0] * U_err_final_qu[mask_p_gt_0])**2
    )
    
    with np.errstate(divide='ignore', invalid='ignore'):
        theta1 = np.rad2deg(np.arctan(np.abs(U_final_qu / Q_final_qu)))
    
    theta_list = []
    for k in range(len(Q_final_qu)):
        q_val, u_val = Q_final_qu[k], U_final_qu[k]
        t1_val = theta1[k]
        
        if q_val == 0:
            theta_list.append(90.0 if u_val >= 0 else 270.0)
            continue

        if u_val >= 0 and q_val > 0:
            theta_list.append(t1_val)
        elif u_val > 0 and q_val < 0:
            theta_list.append(180. - t1_val)
        elif u_val < 0 and q_val < 0:
            theta_list.append(180. + t1_val)
        elif u_val < 0 and q_val > 0:
            theta_list.append(360. - t1_val)

    theta_np = 0.5 * np.array(theta_list)
    PA_deg_plot_qu = np.where(theta_np < 90, theta_np + 180, theta_np)
    
    theta_err = np.zeros_like(p_biased)
    theta_err[mask_p_gt_0] = (90. / np.pi) * (p_biased[mask_p_gt_0]**(-2)) * np.sqrt(
        (Q_final_qu[mask_p_gt_0]**2 * U_err_final_qu[mask_p_gt_0]**2) + (U_final_qu[mask_p_gt_0]**2 * Q_err_final_qu[mask_p_gt_0]**2)
    )
    PA_err_deg_plot_qu = theta_err

    deb_pol = []
    for l in range(len(p_biased)):
        if p_biased[l] > 0 and (p_biased[l] - p_err[l] > 0):
            h = 1.
            deb_pol.append( (p_biased[l] - ((p_err[l]**2)/p_biased[l])) * h )
        else:
            deb_pol.append(0.0)
            
    P_plot_qu = np.array(deb_pol)
    P_err_plot_qu = p_err

    if len(wave_final_qu) > 0:
        ax2.step(wave_final_qu, Q_final_qu * 100., color='k', where="mid", lw=1.0)
        ax3.step(wave_final_qu, U_final_qu * 100., color='k', where="mid", lw=1.0)
        ax4.step(wave_final_qu, P_plot_qu * 100., color='k', where="mid", lw=1.0)
        ax5.plot(wave_final_qu, PA_deg_plot_qu, 'k.')
        
        ax2.fill_between(wave_final_qu, (Q_final_qu - Q_err_final_qu) * 100., (Q_final_qu + Q_err_final_qu) * 100., step="mid", color='k', alpha=0.25)
        ax3.fill_between(wave_final_qu, (U_final_qu - U_err_final_qu) * 100., (U_final_qu + U_err_final_qu) * 100., step="mid", color='k', alpha=0.25)
        ax4.fill_between(wave_final_qu, (P_plot_qu - P_err_plot_qu) * 100., (P_plot_qu + P_err_plot_qu) * 100., step="mid", color='k', alpha=0.25)
        ax5.errorbar(wave_final_qu, PA_deg_plot_qu, yerr=PA_err_deg_plot_qu, fmt='none', color='k', elinewidth=0.7, capsize=1.5, alpha=0.7)

        for ax_pol in [ax2, ax3, ax4]: ax_pol.axhline(0.0, color="k", ls="--", alpha=0.4, lw=0.8)
        
        plot_file_qu_final = f"{q_u_obj_name}_{q_u_epoch_name}_QUPA_binned{q_u_wave_binning}A.pdf"
        plt.savefig(plot_file_qu_final)
        plt.show()

        pol_dat_filename_qu = f"{q_u_obj_name}_{q_u_epoch_name}_specpol_binned{q_u_wave_binning}A.dat"
        pol_table = Table([wave_final_qu, Q_final_qu*100, Q_err_final_qu*100, U_final_qu*100, U_err_final_qu*100, P_plot_qu*100, P_err_plot_qu*100, PA_deg_plot_qu, PA_err_deg_plot_qu],
                          names=('WAVELENGTH_REST', 'Q_PERCENT', 'Q_ERR_PERCENT', 'U_PERCENT', 'U_ERR_PERCENT', 'P_DEBIASED_PERCENT', 'P_ERR_PERCENT', 'PA_DEG', 'PA_ERR_DEG'))
        ascii.write(pol_table, pol_dat_filename_qu, overwrite=True, format='ecsv')
        print(f"Final Q/U polarimetry data written to: {pol_dat_filename_qu}")

        flux_dat_filename_qu = f"{q_u_obj_name}_{q_u_epoch_name}_total_flux_binned{q_u_wave_binning}A.dat"
        flux_table = Table([wave_final_qu, flux_for_dat_qu, flux_err_for_dat_qu], names=("WAVELENGTH_REST", "FLUX_REBINNED", "FLUX_ERR_REBINNED"))
        ascii.write(flux_table, flux_dat_filename_qu, overwrite=True, format='ecsv')
        print(f"Final flux data written to: {flux_dat_filename_qu}")
    else:
        print("WARNING: Q/U plotting - Final wavelength array for plotting was empty.")
        
    print("=" * 50)

# --- Main Processing Function ---
def get_specpol_interactive():
    print("Starting Interactive Spectral Polarimetry Analysis")
    print("=" * 50)

    obj_name_base = get_string("Enter the base name for the target object (e.g., sn2023ixf)", allow_empty=False)
    null_std_name = get_string("Enter the base name for the NULL standard (e.g., hd154892)", allow_empty=False)
    null_std_filter_name = null_std_name + "filter"

    num_pol_stds = get_int("How many polarization standards do you have? (e.g., 1)", default=1)
    pol_std_names = [get_string(f"Enter base name for polarization standard {i+1} (e.g., hd155528)", allow_empty=False) for i in range(num_pol_stds)]
    num_obj_loops = get_int("How many observation loops for the target object? (e.g., 3)", default=1)

    low_wave = get_float("Enter lower wavelength limit for trimming", default=4550)
    high_wave = get_float("Enter upper wavelength limit for trimming", default=9000)

    do_gw = get_yes_no("Apply corrective g and w factors (do_gw)?")
    do_rebin_stokes_global = get_yes_no("Rebin P% when plotting Stokes results (do_rebin)?")
    binning_stokes_global = get_int("Bin size for Stokes processing (binning, e.g., 50)?", default=50)
    n_old_stokes_global = get_float("Original dispersion (n_old, e.g., 2.0)?", default=2.0)
    debias_stokes_global = get_yes_no("Debias P% (debias)?")
    flipq_stokes_global = get_yes_no("Flip the sign of Q (flipq)?") 
    correct_pa_stokes_global_preference = get_yes_no("Correct for instrumental PA (correct_pa)?") 

    filnull_raw_traces = read_trace_files(null_std_filter_name)
    null_raw_traces = read_trace_files(null_std_name)
    pol_stds_raw_traces = [read_trace_files(name) for name in pol_std_names if name]
    
    obj_loops_raw_traces = []
    for i in range(num_obj_loops): 
        current_loop_suffix = ""
        if num_obj_loops > 1:
            current_loop_suffix = f"_lp{i+1}"
        
        traces = read_trace_files(obj_name_base, loop_suffix_from_input=current_loop_suffix)
        if traces: obj_loops_raw_traces.append(traces)

    if not filnull_raw_traces or not null_raw_traces or any(tr is None for tr in pol_stds_raw_traces) or any(tr is None for tr in obj_loops_raw_traces):
        print("Critical error: Could not read all necessary data files. Exiting.")
        return

    filnull_traces = sp.trim(filnull_raw_traces, low_wave, high_wave)
    null_traces = sp.trim(null_raw_traces, low_wave, high_wave)
    pol_stds_traces = [sp.trim(tr, low_wave, high_wave) for tr in pol_stds_raw_traces]
    obj_loops_traces = [sp.trim(tr, low_wave, high_wave) for tr in obj_loops_raw_traces]

    print("\n--- Processing Filter Null Standard (Polarizance Test) ---")
    fname_filnull = null_std_filter_name + "_interactive"
    filnull_debias_choice = get_yes_no("Debias for Filter Null (default: False)?")
    filnull_binning_choice = get_int("Binning for Filter Null (default: 200)?", default=200)
    filnull_correct_pa_choice = get_yes_no("Correct PA for Filter Null (default: False)?")
    filnull_flipq_choice = get_yes_no("Flip Q for Filter Null (default: True)?")

    inst_pa_dummy = [] 
    inst_pa_calculated_for_correction = None 

    try:
        filnull_results = sp.stokes_and_pol(filnull_traces, inst_pa_dummy, fname_filnull,
                                            polarizance=True, do_gw=do_gw, do_rebin=do_rebin_stokes_global, 
                                            binning=filnull_binning_choice, n_old=n_old_stokes_global,
                                            debias=filnull_debias_choice, flipq=filnull_flipq_choice, 
                                            correct_pa=filnull_correct_pa_choice)
        
        if filnull_results and len(filnull_results) > 7 and filnull_results[0] is not None : 
            sp.write_stokes(filnull_results)
            waves = filnull_results[0]
            if filnull_results[7] is not None and np.array(filnull_results[7]).size > 0:
                pa_instrumental_raw = np.array(filnull_results[7]) 
                kernel_sz = min(99, pa_instrumental_raw.size - (1 if pa_instrumental_raw.size % 2 == 0 else 0))
                if kernel_sz < 1: kernel_sz = 1
                
                inst_pa_calculated_for_correction = signal.medfilt(pa_instrumental_raw, kernel_size=kernel_sz)
                plt.figure(); plt.title(f"Smoothed Instrumental PA from {null_std_filter_name}")
                plt.plot(waves, pa_instrumental_raw, label="Raw PA_I", alpha=0.5)
                plt.plot(waves, inst_pa_calculated_for_correction, label=f"Smoothed PA_I (medfilt {kernel_sz})")
                plt.xlabel("Wavelength (Angstrom)"); plt.ylabel("Instrumental PA (degrees)")
                plt.legend(); plt.grid(True, alpha=0.3); plt.show()
            else:
                print("WARNING: PA_I from filter null (filnull_results[7]) is None or empty. Cannot derive master instrumental PA.")
        else: 
            print(f"Error: Failed to process filter null standard {null_std_filter_name} or PA_I not found/valid in results.")
            print("Exiting due to failed filter null processing.") 
            return 
    except Exception as e:
        print(f"Exception during filter null processing: {e}")
        print("Exiting due to exception in filter null processing.") 
        return
        
    def call_spolpat3_for_target(target_name_str, traces_data, global_n_old, global_binning, global_debias, global_flipq, global_correct_pa_pref, derived_inst_pa):
        fname_base = f"{target_name_str}_interactive" 

        target_binning = get_int(f"Binning for {target_name_str} (- plotting purposes only!)?", default=global_binning)
        
        effective_correct_pa = global_correct_pa_pref
        pa_arg_for_spolpat = [] 

        if global_correct_pa_pref:
            if derived_inst_pa is not None and derived_inst_pa.size > 0:
                pa_arg_for_spolpat = derived_inst_pa
            else:
                print(f"WARNING: Instrumental PA correction for {fname_base} SKIPPED (PA data not available).")
                effective_correct_pa = False
        
        try:
            target_results = sp.stokes_and_pol(traces_data, pa_arg_for_spolpat, fname_base, 
                                             polarizance=False,
                                             do_gw=do_gw,
                                             do_rebin=do_rebin_stokes_global,
                                             binning=target_binning,
                                             n_old=global_n_old,
                                             debias=global_debias,
                                             flipq=global_flipq, 
                                             correct_pa=effective_correct_pa)
            if target_results:
                sp.write_stokes(target_results) 
                if get_yes_no(f"View all traces for {target_name_str}?"):
                    sp.see_all_traces(traces_data, 0.0, side="top")
                    sp.see_all_traces(traces_data, 0.0, side="bottom")
                if get_yes_no(f"Plot q-u for {target_name_str} ?"):
                    plt.figure(); plt.title(f"Q-U for {fname_base} (spolpat3 output)")
                    plt.plot(target_results[0], target_results[1], label="q (frac)")
                    plt.plot(target_results[0], target_results[2], label="u (frac)")
                    plt.xlabel("Wavelength (Angstrom)"); plt.ylabel("Stokes Q,U (fractional)")
                    plt.legend(); plt.grid(True, alpha=0.3); plt.show()
                return target_results
            else: print(f"Error: Failed to process {fname_base}.")
        except Exception as e_target:
            print(f"Exception during {fname_base} processing: {e_target}")
        return None

    print("\n--- Processing Null Standard ---")
    null_results = call_spolpat3_for_target(null_std_name, null_traces, 
                                            n_old_stokes_global, binning_stokes_global, debias_stokes_global, 
                                            flipq_stokes_global, correct_pa_stokes_global_preference, 
                                            inst_pa_calculated_for_correction)

    print("\n--- Processing Polarization Standard(s) ---")
    pol_stds_results_list = []
    for i, name in enumerate(pol_std_names):
        print(f"\nProcessing Polarization Standard: {name}")
        res = call_spolpat3_for_target(name, pol_stds_traces[i],
                                       n_old_stokes_global, binning_stokes_global, debias_stokes_global,
                                       flipq_stokes_global, correct_pa_stokes_global_preference,
                                       inst_pa_calculated_for_correction)
        if res: pol_stds_results_list.append(res)

    print("\n--- Processing Target Object Loop(s) ---")
    obj_ind_results_list = []
    obj_loops_binning_default = 30 #get_int(f"Default P-binning for object {obj_name_base} loops ?", default=30) #useless????

    for i in range(num_obj_loops):
        res = call_spolpat3_for_target(f"{obj_name_base}_loop{i+1}", obj_loops_traces[i],
                                       n_old_stokes_global, obj_loops_binning_default, 
                                       debias_stokes_global, flipq_stokes_global, 
                                       correct_pa_stokes_global_preference, inst_pa_calculated_for_correction)
        if res: obj_ind_results_list.append(res)

    if not obj_ind_results_list: 
        print("No object loop results to combine/process. Exiting.")
        return

    final_combined_or_single_loop_results = None
    if num_obj_loops > 1:
        print("\n--- Combining Object Loop(s) ---")
        combine_debias = get_yes_no("Debias when combining loops? (default: True)")
        combine_do_cov = get_yes_no("Account for covariance when combining loops? (default: True)")
        combine_do_rebin = get_yes_no("Rebin when combining loops? (default: False)")
        combine_binning = get_int("Binning size if rebinning combined data (default: 50)?", default=50)
        combine_n_old = get_float("Original binning/pixel size for combined rebinning (default: 2.0)?", default=n_old_stokes_global)
        
        try:
            final_combined_or_single_loop_results = sp.combine_loops(obj_ind_results_list,
                                                 debias=combine_debias, do_cov=combine_do_cov,
                                                 do_rebin=combine_do_rebin, binning=combine_binning,
                                                 n_old=combine_n_old, probe=False, probe_qu=None) 
            if not final_combined_or_single_loop_results:
                print(f"Error: Failed to combine object loops for {obj_name_base}.")
                return
        except Exception as e:
            print(f"Exception during object loop combination: {e}")
            return
    elif obj_ind_results_list: 
        final_combined_or_single_loop_results = obj_ind_results_list[0]

    if final_combined_or_single_loop_results:
        print(f"\n--- Final Results for {obj_name_base} (from original get_specpol processing) ---")
        output_filename_base = get_string("Enter base for final output filename (e.g., sn2023ixf_final)", default=f"{obj_name_base}_final_interactive", allow_empty=False)

        data_to_write = list(final_combined_or_single_loop_results)
        if len(data_to_write) > 8: 
            data_to_write[5] = np.array(data_to_write[5]) * 100.0 
            data_to_write[6] = np.array(data_to_write[6]) * 100.0
            data_to_write[7] = np.rad2deg(np.array(data_to_write[7]))
            data_to_write[8] = np.rad2deg(np.array(data_to_write[8]))
        
        sp.write_stokes(tuple(data_to_write))
        print(f"Final Stokes parameters written by sp.write_stokes (typically to pol_reduction.txt).")

        plt.figure(figsize=(10, 6))
        plt.title(f"Final Stokes Parameters for {obj_name_base} (sp.write_stokes data)")
        plt.plot(final_combined_or_single_loop_results[0], final_combined_or_single_loop_results[1], alpha=0.7, label="q (frac)")
        plt.plot(final_combined_or_single_loop_results[0], final_combined_or_single_loop_results[2], alpha=0.7, label="u (frac)")
        if len(data_to_write) > 5: plt.plot(final_combined_or_single_loop_results[0], data_to_write[5], alpha=0.7, label="P (%)") 
        
        all_vals = np.concatenate((final_combined_or_single_loop_results[1], final_combined_or_single_loop_results[2], data_to_write[5]/100.0 if len(data_to_write)>5 else [] ))
        all_vals_finite = all_vals[np.isfinite(all_vals)]
        if len(all_vals_finite)>0:
            all_min = np.min(all_vals_finite); all_max = np.max(all_vals_finite)
            padding = (all_max - all_min) * 0.1 if (all_max - all_min) > 0 else 0.1
            plt.ylim(all_min - padding, all_max + padding)
        else: plt.ylim(-0.05, 0.05) 

        plt.xlabel("Wavelength (Angstrom)"); plt.ylabel("Value (Q,U frac; P %)")
        plt.legend(); plt.grid(True,alpha=0.3)
        plot_filename = f"{output_filename_base}_stokes_P.pdf" 
        plt.savefig(plot_filename); plt.show()
        
        run_qu_plotting_and_export(final_combined_or_single_loop_results, 
                                   obj_name_base, 
                                   n_old_stokes_global,
                                   flipq_stokes_global) 
    else:
        print("No final results to plot or write from original get_specpol logic.")

    print("\n" + "=" * 50)
    print("Interactive Spectral Polarimetry Analysis Finished.")
    print("=" * 50)

if __name__ == "__main__":
    try:
        get_specpol_interactive()
    except Exception as e_main:
        print(f"CRITICAL ERROR in script execution: {e_main}")
        import traceback
        traceback.print_exc()
