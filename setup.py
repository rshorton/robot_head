from setuptools import setup
from glob import glob

package_name = 'robot_head'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shorton',
    maintainer_email='horton.rscotti@gmail.com',
    description='Robot Head Control Node',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision = robot_head.vision:main',
            'tracker = robot_head.tracker:main',
            'viewer = robot_head.viewer:main'
        ],
    },
    scripts=['robot_head/pose_interp.py',
             'robot_head/BlazeposeDepthai.py',
             'robot_head/mediapipe_utils.py']
)
