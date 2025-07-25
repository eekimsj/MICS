from setuptools import find_packages, setup

package_name = 'dexhand_angle'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dyros',
    maintainer_email='dyros@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'dexhand_angle_pub_node = dexhand_angle.dexhand_angle_pub:main',
        'dexhand_pub_1_1 = dexhand_angle.dexhand_pub_1_1:main',
        'dexhand_publisher = dexhand_angle.modelToDexhand_v3:main',
        ],
    },
)
