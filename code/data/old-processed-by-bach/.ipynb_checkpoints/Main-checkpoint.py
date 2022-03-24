import numpy as np

if __name__ == '__main__':
    in_dir = '/local/scratch/FishData/'
    names = ['WorkingStd1.txt', 'WorkingStd2.txt', 'WorkingStd3.txt', 'WorkingStd4.txt', 'WorkingStd5.txt']
    out_dir = '/home/nguyenhoai2/Research/PycharmProjects/ExtractFishData/ProcessOutput/'

    for name in names:
        # to store data
        masses = []
        time_stamps = []
        intensities = []
        smallest_no_packets = float('inf')

        with open(in_dir+name, 'r') as f:
            lines = f.readlines()
            idx = 0
            while idx < len(lines):
                line = lines[idx]
                if line.startswith('ScanHeader'):
                    intensity = []
                    mass = []
                    # mote to start_time
                    while not lines[idx].startswith('start_time'):
                        idx = idx + 1
                    time_stamps.append(float(lines[idx].split(', ')[0].split(' = ')[1]))
                    # move to the packet
                    while idx < len(lines) and not lines[idx].startswith('Packet'):
                        idx = idx + 1
                    while idx < len(lines) and lines[idx].startswith('Packet'):
                        split = lines[idx].split(', ')
                        p_intensity = float(split[1].split(' = ')[1])
                        p_mass = float(split[2].split(' = ')[1])
                        intensity.append(p_intensity)
                        mass.append(p_mass)
                        idx = idx+3

                    if smallest_no_packets > len(intensity):
                        smallest_no_packets = len(intensity)
                    intensities.append(intensity)
                    masses.append(mass)

                idx = idx+1

        # cut down all the data that are more than the minimum number of packages
        masses = [mass[:smallest_no_packets] for mass in masses]
        ave_mass = np.average(masses, axis=0)
        intensities = [inten[:smallest_no_packets] for inten in intensities]

        idx = 0
        step = 30
        ave_intensities = []
        ave_time_stamps = []
        while (idx+step) < len(masses):
            ave_intensities.append(np.average(intensities[idx:idx+step], axis=0))
            ave_time_stamps.append(np.average(time_stamps[idx:idx+step]))
            idx = idx+step

        ave_intensities.append(np.average(intensities[idx:], axis=0))
        ave_time_stamps.append(np.average(time_stamps[idx:]))

        # output
        out_file = open(out_dir+name, 'w')
        out_file.write('Mass/position\t' + '\t'.join([str(m) for m in ave_mass])+'\n')
        out_file.write('Start time\n')
        for time, intensity in zip(ave_time_stamps, ave_intensities):
            out_file.write(str(time)+'\t'+'\t'.join([str(intent) for intent in intensity])+'\n')
        out_file.close()
    print('Finish')

