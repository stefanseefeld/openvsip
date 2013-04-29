from qm.fields import TextField, SetField
from qm.test.classes import ssh_host
import sys
import os

class RemoteHost(ssh_host.SSHHost):

    # The wrapper script name to be used to invoke
    # remote programs
    wrapper = TextField()
    wrapper_args = SetField(TextField())

    def Run(self, path, arguments, environment = None, timeout = -1,
            relative = False):
        
        if self.wrapper:
            path = os.path.join(os.curdir, os.path.basename(path))
            arguments = [path] + arguments[:]
            path = self.wrapper
            if self.wrapper_args:
                arguments = self.wrapper_args + arguments[:]
        return super(RemoteHost, self).Run(path, arguments,
                                           environment, timeout, relative)

    def DeleteFile(self, remote_file):

        if self.default_dir:
            remote_file = os.path.join(self.default_dir, remote_file)
        super(RemoteHost, self).Run('rm', [remote_file])
