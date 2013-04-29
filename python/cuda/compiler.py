import subprocess, hashlib, tempfile, os

class CompileError(Exception):
    def __init__(self, msg, command_line, stdout=None, stderr=None):
        self.msg = msg
        self.command_line = command_line
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self):
        result = self.msg
        if self.command_line:
            try:
                result += "\n[command: %s]" % (" ".join(self.command_line))
            except Exception, e:
                print e
        if self.stdout:
            result += "\n[stdout:\n%s]" % self.stdout
        if self.stderr:
            result += "\n[stderr:\n%s]" % self.stderr

        return result


def get_nvcc_version(nvcc):

    cmdline = [nvcc, '--version']
    try:
        p = subprocess.Popen(cmdline,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             close_fds=True)
        stdout, stderr = p.communicate()
        return stdout
    except OSError, e:
        raise OSError("%s was not found (is it on the PATH?) [%s]" % (nvcc, str(e)))


def compile_plain(source, options, keep, nvcc, cache_dir):

    if cache_dir:
        checksum = hashlib.md5()

        checksum.update(source)
        for option in options: 
            checksum.update(option)
        checksum.update(get_nvcc_version(nvcc))

        cache_file = checksum.hexdigest()
        cache_path = os.path.join(cache_dir, cache_file + ".cubin")

        try:
            return open(cache_path, "rb").read()
        except:
            pass

    file_dir = tempfile.mkdtemp()
    file_root = "kernel"

    cu_file_name = file_root + ".cu"
    cu_file_path = os.path.join(file_dir, cu_file_name)

    outf = open(cu_file_path, "w")
    outf.write(str(source))
    outf.close()

    if keep:
        options = options[:]
        options.append("--keep")

        print "*** compiler output in %s" % file_dir

    cmdline = [nvcc, "--cubin"] + options + [cu_file_name]
    p = subprocess.Popen(cmdline, cwd=file_dir,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         close_fds=True)
    stdout, stderr = p.communicate()

    try:
        cubin_f = open(os.path.join(file_dir, file_root + ".cubin"), "rb")
    except IOError:
        no_output = True
    else:
        no_output = False

    if no_output and (stdout or stderr):
        raise CompileError("nvcc compilation of %s failed" % cu_file_path,
                           cmdline, stdout=stdout, stderr=stderr)

    if stdout or stderr:
        from warnings import warn
        warn("The CUDA compiler suceeded, but said the following:\n"
             +stdout+stderr)

    cubin = cubin_f.read()
    cubin_f.close()

    if cache_dir:
        outf = open(cache_path, "wb")
        outf.write(cubin)
        outf.close()

    if not keep:
        from os import listdir, unlink, rmdir
        for name in listdir(file_dir):
            unlink(os.path.join(file_dir, name))
        rmdir(file_dir)

    return cubin


def _get_per_user_string():
    try:
        from os import getuid
    except ImportError:
        checksum = hashlib.md5()
        from os import environ
        checksum.update(environ["HOME"])
        return checksum.hexdigest()
    else:
        return "uid%d" % getuid()


def compile(source, nvcc='nvcc', options=[], keep=False,
            no_extern_c=False, arch=None, code=None, cache_dir=None,
            include_dirs=[]):

    if not no_extern_c:
        source = 'extern "C" {\n%s\n}\n' % source

    options = options[:]
    if arch is None:
        try:
            import device
            arch = "sm_%d%d" % device.get_device().compute_capability()
        except RuntimeError:
            pass

    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), 
                "svxx-cuda-compiler-cache-v1-%s" % _get_per_user_string())

        from os import mkdir
        try:
            mkdir(cache_dir)
        except OSError, e:
            from errno import EEXIST
            if e.errno != EEXIST:
                raise

    if arch is not None:
        options.extend(["-arch", arch])

    if code is not None:
        options.extend(["-code", code])

    for i in include_dirs:
        options.append("-I"+i)

    return compile_plain(source, options, keep, nvcc, cache_dir)


class SourceModule(object):
    def __init__(self, source, nvcc='nvcc', options=[], keep=False,
                 no_extern_c=False, arch=None, code=None, cache_dir=None,
                 include_dirs=[]):
        self._check_arch(arch)

        cubin = compile(source, nvcc, options, keep, no_extern_c, 
                        arch, code, cache_dir, include_dirs)

        import module
        self.module = module.Module(cubin)

        self.get_global = self.module.get_global
        #self.get_texref = self.module.get_texref

    def _check_arch(self, arch):
        if arch is None: return
        try:
            import device
            capability = device.get_device().compute_capability()
            if tuple(map(int, tuple(arch.split("_")[1]))) > capability:
                from warnings import warn
                warn("trying to compile for a compute capability "
                        "higher than selected GPU")
        except:
            pass

    def get_function(self, name):
        return self.module.get_function(name)
