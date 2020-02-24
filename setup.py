import os
from setuptools import setup


# Load package info, without importing the package
basedir = os.path.dirname(os.path.abspath(__file__))
package_info_path = os.path.join(basedir, "dwave", "inspector", "package_info.py")
package_info = {}
with open(package_info_path, encoding='utf-8') as f:
    exec(f.read(), package_info)


# Package requirements, minimal pinning
install_requires = [
    'dimod>=0.8.17',
    'dwave-system>=0.8.1',
    'dwave-cloud-client>=0.6.3',
    'Flask>=1.1.1',
    # dwave-inspectorapp==0.1.3
]

# Package extras requirements
extras_require = {
    'test': ['coverage', 'mock', 'vcrpy'],

    # backports
    ':python_version < "3.7"': ['importlib_resources']
}

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
]

packages = ['dwave', 'dwave.inspector']

python_requires = '>=3.5'

setup(
    name=package_info['__package_name__'],
    version=package_info['__version__'],
    author=package_info['__author__'],
    author_email=package_info['__author_email__'],
    description=package_info['__description__'],
    long_description=open('README.rst', encoding='utf-8').read(),
    url=package_info['__url__'],
    license=package_info['__license__'],
    packages=packages,
    entry_points={
        'inspectorapp_viewers': [
            'browser_tab = dwave.inspector.viewers:webbrowser_tab',
            'browser_window = dwave.inspector.viewers:webbrowser_window',
        ],
        'dwave_contrib': [
            'dwave-inspector = dwave.inspector.package_info:contrib'
        ]
    },
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=classifiers,
    zip_safe=False,
)
