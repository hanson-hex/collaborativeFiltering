  update [dbo].[t_AutoCoding_ViewEntity] set label = 'Permission', LabelForPlural = 'Permissions' where name = 'permission'
  update [dbo].[t_AutoCoding_Entity] set isRoot = 0 where UniqueName = 'services'

  update [dbo].[t_AutoCoding_FieldDocExtension] set Description = 'Specific email addresses that offline message are sent to. Available and required when Offline Message Mail Type is "toEmailAddresses".' where FieldId = 'C5B25329-7C46-EB11-8100-00155D081D0B'

