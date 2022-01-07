#%%
# Start the solver by running this script (both for a Python shell and Jupyter).
# It check the OS type and wraps the solver call approrpiately for Windows (bug with multiprocessing library).
import os

if os.name == 'nt':
    if __name__ == "__main__":
        import lrz_data_solver_threaded
else:
    import lrz_data_solver_threaded

# %%
