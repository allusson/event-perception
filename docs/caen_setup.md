# Working on a CAEN lab PC

## The idea

Three things are needed to work on this project:

1. **The code.** Lives in `N:\event-perception`. N:\ is your network drive, so it is safe and follows you between machines.
2. **Git.** The tool that downloads the code from GitHub and uploads your changes.
3. **The venv.** A private box of Python packages for this project. It lives on C:\ because N:\ is far too slow for it.

C:\ may be erased between logins. That is fine: nothing on C:\ is precious. If Git, uv or the venv are gone, you reinstall them with the commands below in a couple of minutes. Your code and data on N:\ are untouched.

## Each time you sit down

Open PowerShell and run these three lines:

```powershell
cd N:\event-perception
& $env:LOCALAPPDATA\ep-venv\Scripts\activate.ps1
git pull
```

Your prompt should now start with `(ep-venv)`. You are ready to work. Start notebooks with `jupyter lab`.

If a line fails, find the matching fix here, then run the three lines again:

| What you see | What it means | Fix |
| --- | --- | --- |
| `activate.ps1 ... not recognized` | The venv was erased | "Rebuild the venv" below |
| `uv ... not recognized` | uv was erased | "Install uv" below |
| `git ... not recognized` | Git was erased | "Install Git" below |

## Before you leave

```powershell
git add -A
git commit -m "describe what you changed"
git push
```

The first push opens a browser to sign in to GitHub. It's a shared computer, so sign out of GitHub in the browser when you're done.

## The fixes

After installing uv or Git, **close PowerShell and open a new window**. A window only notices new programs when it starts.

**Install uv** (the tool that installs Python and packages):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Install Git:**

```powershell
winget install --id Git.Git --scope user
```

**Rebuild the venv** (takes a minute or two):

```powershell
cd N:\event-perception
uv venv $env:LOCALAPPDATA\ep-venv --python 3.12
& $env:LOCALAPPDATA\ep-venv\Scripts\activate.ps1
uv pip install -e .
uv pip install jupyter
python -c "import evcam; print('ok')"
```

The last line should print `ok`.

## First time on a new CAEN account

Only needed once, because the result lives on N:\.

1. Install uv and Git (above), then open a new PowerShell window.
2. Download the code and set your name for commits:

   ```powershell
   N:
   git clone https://github.com/allusson/event-perception.git
   cd event-perception
   git config --local user.name "Raphael Allusson"
   git config --local user.email "allusson@umich.edu"
   ```

3. Rebuild the venv (above).

## Good to know

- **Data and model weights are not on GitHub.** Copy recordings into `data\` yourself. Public datasets come from `python scripts\download_data.py`.
- **Extras.** For notebook 01 with N-MNIST, run the tonic install line from that notebook. For notebook 03 (EDI, EFNet), run `uv pip install -e ".[deblur]"`, which is a large download.
- **Don't put the venv or Python on N:\.** It was tried and took over ten minutes without finishing. You also can't create folders directly in `C:\`, which is why the venv goes in `$env:LOCALAPPDATA` (your personal folder on C:\).
- **Pull first, push last.** If you edit the same notebook on two computers without syncing, Git can't merge them cleanly.
- **After recording in DV,** load the clip with `evcam.io.load_aedat4` before leaving the lab and confirm it has real APS frames, not Accumulator output.
