// Contravía's launcher: one file per OS. The Python runtime and the game ride at its end as a zip; the
// first run of a version unpacks them into the per-user data folder, and every run starts the game there.
// Arguments go to `contravia` as they are; with none, it plays.
//
// Layout of this file: the launcher, then the zip, then the zip's length as 8 little-endian bytes.
package main

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// version names the unpacked runtime's folder; the build sets it to the game's commit.
var version = "dev"

func dataHome() string {
	if h := os.Getenv("CONTRAVIA_HOME"); h != "" {
		return h
	}
	if runtime.GOOS == "windows" {
		if h := os.Getenv("LOCALAPPDATA"); h != "" {
			return filepath.Join(h, "Contravia")
		}
	}
	if h := os.Getenv("XDG_DATA_HOME"); h != "" {
		return filepath.Join(h, "contravia")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "share", "contravia")
}

func payload(exe string) (*zip.Reader, *os.File, error) {
	f, err := os.Open(exe)
	if err != nil {
		return nil, nil, err
	}
	st, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	tail := make([]byte, 8)
	if _, err := f.ReadAt(tail, st.Size()-8); err != nil {
		return nil, nil, err
	}
	n := int64(binary.LittleEndian.Uint64(tail))
	if n <= 0 || n > st.Size()-8 {
		return nil, nil, fmt.Errorf("%s carries no game", exe)
	}
	z, err := zip.NewReader(io.NewSectionReader(f, st.Size()-8-n, n), n)
	return z, f, err
}

// unpack writes the zip into dest through a temporary folder, so a dest that exists is whole.
func unpack(z *zip.Reader, dest string) error {
	tmp := dest + ".part"
	os.RemoveAll(tmp)
	for _, zf := range z.File {
		p := filepath.Join(tmp, filepath.FromSlash(zf.Name))
		if !strings.HasPrefix(p, tmp+string(os.PathSeparator)) {
			return fmt.Errorf("bad path in payload: %s", zf.Name)
		}
		if zf.FileInfo().IsDir() {
			if err := os.MkdirAll(p, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
		if err := write(zf, p); err != nil {
			return err
		}
	}
	return os.Rename(tmp, dest)
}

func write(zf *zip.File, p string) error {
	r, err := zf.Open()
	if err != nil {
		return err
	}
	defer r.Close()
	mode := zf.Mode().Perm() | 0o600
	w, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	if _, err := io.Copy(w, r); err != nil {
		w.Close()
		return err
	}
	return w.Close()
}

// prune removes the runtimes of other versions, so an update does not keep the old one on disk.
func prune(root, keep string) {
	entries, _ := os.ReadDir(root)
	for _, e := range entries {
		if e.Name() != keep {
			os.RemoveAll(filepath.Join(root, e.Name()))
		}
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "contravia:", err)
	os.Exit(1)
}

func main() {
	exe, err := os.Executable()
	if err != nil {
		fail(err)
	}
	root := filepath.Join(dataHome(), "runtime")
	dest := filepath.Join(root, version)
	if _, err := os.Stat(dest); err != nil {
		z, f, err := payload(exe)
		if err != nil {
			fail(err)
		}
		fmt.Println("Contravía: preparando el juego por primera vez...")
		if err := os.MkdirAll(root, 0o755); err != nil {
			fail(err)
		}
		err = unpack(z, dest)
		f.Close()
		if err != nil {
			fail(err)
		}
		prune(root, version)
	}
	python := filepath.Join(dest, "python", "bin", "python3.14t")
	if runtime.GOOS == "windows" {
		python = filepath.Join(dest, "python", "pythonw.exe")
	}
	args := os.Args[1:]
	if len(args) == 0 {
		args = []string{"play"}
	}
	cmd := exec.Command(python, append([]string{"-m", "contravia.cli"}, args...)...)
	cmd.Dir = filepath.Join(dest, "app")
	cmd.Env = append(os.Environ(), "PYTHONPATH="+filepath.Join(dest, "app")+string(os.PathListSeparator)+
		filepath.Join(dest, "site"), "PYTHONNOUSERSITE=1")
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			os.Exit(ee.ExitCode())
		}
		fail(err)
	}
}
