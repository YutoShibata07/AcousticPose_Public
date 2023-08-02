function bvh2csv_batch

    dir_name = '~/Downloads/bvh_20211005/';
    list = dir(sprintf('%s/*.bvh', dir_name));

    for i = 1 : size(list, 1)
        [filepath, name, ext] = fileparts(list(i).name);
        bvh_name = sprintf('%s/%s.bvh', list(i).folder, name);
        csv_name = sprintf('%s/%s_position.csv', list(i).folder, name);
        bvh2csv(bvh_name, csv_name)
    end
end